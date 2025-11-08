#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.08 #
#####################################################
import time, torch
import sys

class loss_cure():
    def __init__(self, net, criterion, lambda_, device='cuda'):
        self.net = net
        self.criterion = criterion
        self.lambda_ = lambda_  # 1
        self.device = device

    def _find_z(self, inputs, targets, h):
        # 该函数主要计算一个归一化的扰动向量 𝑧, 用于对输入数据施加扰动, 并返回其范数 norm_grad, 通常用于评估模型的鲁棒性（例如对抗训练或鲁棒优化）
        # print("inputs.size: ",inputs.size()) # torch.Size([8, 3, 32, 32])

        inputs.requires_grad_()
        outputs = self.net.eval()(inputs)  # 一个完整的分类模型的输入和输出
        loss_z = self.criterion(outputs, targets) # self.net.eval()(inputs)
        loss_z.backward()                  # torch.ones(targets.size(), dtype=torch.float).to(self.device)
        grad = inputs.grad.data + 0.0      # 形状相同的张量
        norm_grad = grad.norm().item()     # 计算L2范数
        z = torch.sign(grad).detach() + 0. # z是自我构造的归一化扰动 （使用输入的梯度信息生成扰动方向，类似FGSM）
        z = 1. * (h) * (z + 1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None] + 1e-7)
        inputs.grad.detach()
        inputs.grad.zero_()
        # zero_gradients(inputs)
        self.net.zero_grad()

        return z, norm_grad

    def regularizer(self, inputs, targets, h=3., lambda_=4):
        '''
        Regularizer term in CURE： 它通过对比扰动前后的损失差异及其对输入的梯度影响，计算出一个正则化值。
        '''
        # 生成一个基于梯度方向的扰动z，生成幅度 h
        z, norm_grad = self._find_z(inputs, targets, h)
        # print("regularizer: ", z.size(), norm_grad)  # 扰动 + 梯度的L2范数

        inputs.requires_grad_()
        outputs_pos = self.net.eval()(inputs + z)
        outputs_orig = self.net.eval()(inputs)

        loss_pos = self.criterion(outputs_pos, targets)
        loss_orig = self.criterion(outputs_orig, targets)

        # 两个相同样本, 但是一个攻击后loss + 没攻击的loss, 返回两者之间的梯度差距
        grad_diff = torch.autograd.grad((loss_pos - loss_orig), inputs)[0] # torch.Size([8, 3, 32, 32])
        reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)         # reg: 每个样本梯度差距的L2范数 # 8个样本
        self.net.zero_grad()

        return torch.sum(self.lambda_ * reg) / float(inputs.size(0)), norm_grad  # 每个样本的平均梯度差值


def procedure(train_loader_1, train_loader_2, network, criterion, scheduler, optimizer, mode, grad=False, h=3.0):
    # mode 默认是 train
  # losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
  if mode == 'train'  : network.train()
  elif mode == 'valid': network.eval()
  else: raise ValueError("The mode is not right : {:}".format(mode))
  grads = {}
  # data_time, batch_time, end = AverageMeter(), AverageMeter(), time.time()

  ############################################# adjust h
  # loader2 用于正则化调整
  inputs, targets = next(iter(train_loader_2)) # torch.Size([8, 3, 32, 32])
  inputs = inputs.cuda()
  targets = targets.cuda(non_blocking=True)
  reg = loss_cure(network, criterion, lambda_=1, device='cuda')
  regularizer_average, grad_norm = reg.regularizer(inputs, targets, h = h)
  # print("regularizer_average: ",regularizer_average)  # tensor(4.2430e-05, device='cuda:0')   每个样本，被攻击之后的模型输出平均变化值

  # 50 次 是内部循环所控制，用于计算特征值
  for i, (inputs, targets) in enumerate(train_loader_1):
    # print(inputs.size())
    inputs = inputs.cuda()
    targets = targets.cuda(non_blocking=True)
    if mode != 'train': return 0,0,0,time.time()-time.time()

    logits = network(inputs)
    loss   = criterion(logits, targets)
    # backward
    if mode == 'train':
      loss.backward()
      import copy
      index_grad = 0
      index_name = 0

      # 梯度的提取与存储  内部10次循环
      for name, param in network.named_parameters():
           # print(name) # 似乎只计算了 DARTS cell 0 的前两条边
           if param.grad is None:
                print('param.grad is None')
                print(name)
                continue
           #if param.grad.view(-1)[0] == 0 and param.grad.view(-1)[1] == 0: continue #print(name)
           if index_name > 10: break
           if len(param.grad.view(-1).data[0:100]) < 50: # 将每个梯度平铺为一维， 提取前 100 个元素并存储到grad字典中 (保存10个梯度len为50的模型数据)
               continue
           index_grad = name
           index_name += 1

           if name in grads: # 收集计算的参数
               grads[name].append(copy.copy(param.grad.view(-1).data[0:100])) # 已存储则追加
           else:
               grads[name]=[copy.copy(param.grad.view(-1).data[0:100])]       # 没存储则新建

      if len(grads[index_grad]) == 50:  # 某个操作的梯度值list，长度 =50。
             # print(index_grad)
             conv = 0
             maxconv = 0
             minconv = 0
             lower_layer = 1
             top_layer = 1
             para = 0

             for name in grads:
                # print(len(grads[name]))   # len = 50
                for i in range(50): # nt(self.grads[name][0].size()[0])):
                   #if len(grads[name])!=: print(name)
                   #for j in range(50):
                   #if i == j: continue
                   grad1 = torch.tensor([grads[name][k][i] for k in range(25)])
                   grad2 = torch.tensor([grads[name][k][i] for k in range(25,50)])
                   # grad1 = grad1 - grad1.mean()
                   # grad2 = grad2 - grad2.mean()
                   # print(grad1.size(), grad2.size()) [25], [25]
                   conv += torch.dot(grad1, grad2) / 2500
                   para += 1
             # print(para) 550
             break

    # count time
    # batch_time.update(time.time() - end)
    # end = time.time()

  if mode == 'train':
      # print(conv)
      # sys.exit()
      RF = -torch.exp(conv  * 5000000) * regularizer_average  # conv应该就是特征值  * tensor(4.2430e-05, device='cuda:0')  这块有点像L: |y - f(x)|

      return RF#, 0,0, batch_time.sum #conv, maxconv, minconv

  else:
      return 0#,0,0, batch_time.sum

