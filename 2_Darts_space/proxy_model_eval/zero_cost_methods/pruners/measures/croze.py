import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import copy
from . import measure
from ..p_utils import adj_weights, get_layer_metric_array_adv_feats

def fgsm_attack(net, image, target, epsilon):
    perturbed_image = image.detach().clone()  # 复制输入，防止原数据改变
    perturbed_image.requires_grad = True      # 开启梯度追踪
    net.zero_grad()                           # 清空梯度

    logits = net(perturbed_image)             # 前向传播获取输出   (属于一开始就切好子网的类型)
    loss = F.cross_entropy(logits, target)    # 计算交叉熵损失
    loss.backward()                           # 反向传播获取梯度

    sign_data_grad = perturbed_image.grad.sign_()                 # 提取梯度符号
    perturbed_image = perturbed_image - epsilon * sign_data_grad  # 添加扰动
    perturbed_image = torch.clamp(perturbed_image, 0, 1)          # 限制扰动值在合法范围

    return perturbed_image

# 通过装饰器，compute_synflow_per_weight 被注册到 _measure_impls 字典中，名称为 'croze'

@measure('croze', bn=False, mode='param')
def compute_synflow_per_weight(net, inputs, targets, mode, split_data=1, loss_fn=None, search_space='nasbench201'):

    # print(inputs.size())

    device = inputs.device
    origin_inputs, origin_outputs = inputs, targets
    
    cos_loss = nn.CosineSimilarity(dim=0)
    ce_loss = nn.CrossEntropyLoss()
    
    # convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        # 网络线性化，把网络中的参数变成整数，返回符号
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)  # 保存符号
            param.abs_()                     # 参数绝对值
        return signs

    #convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        # 非线性化：在计算完成后恢复原始权重符号，保证网络的一致性。
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])      # 恢复符号

    advnet = copy.deepcopy(net)

    # keep signs of all params
    signs = linearize(net)         # 保留+-的符号
    adv_signs = linearize(advnet)  # 保留+-的符号
    
    # Compute gradients with input of 1s
    net.zero_grad()
    net.double()                   # 把网络当中所有数值修改为双精度
    advnet.double()                # 主要用于生成对抗样本

    # 输入clean图片
    output, feats = net.forward(origin_inputs.double(), return_activated_feats=True)
    output.retain_grad()           # BP时保留output的梯度

    # 调整权重偏向于抗干扰方向 (给net增加权重漂移，再对其设计对抗攻击样本，若变化小则代表稳定？)
    advnet = adj_weights(advnet, origin_inputs.double(), origin_outputs, 2.0, loss_maximize=True)

    advinput = fgsm_attack(advnet, origin_inputs.double(), origin_outputs, 0.01)  # 图片、标签、eps

    # 这里的 advnet 被简单调整过权重,再输入对抗样本
    advnet.train()
    adv_outputs, adv_feats = advnet.forward(advinput.detach(), return_activated_feats=True)
    adv_outputs.retain_grad()
    # print(len(adv_feats))

    # 两者损失之和代表了原始模型和对抗模型在给定输入上的总体损失
    # loss = ce_loss(output, origin_outputs) + ce_loss(adv_outputs, origin_outputs)        # 居然是1:1
    loss_acc = ce_loss(output, origin_outputs)
    loss_adv = ce_loss(adv_outputs, origin_outputs)
    loss = loss_acc + loss_adv
    # print(f"loss: {loss.item():.3e}, loss_acc: {loss_acc.item():.3e}, loss_adv: {loss_adv.item():.3e}")
    loss.backward()   # 不进行反向传播，无法收集数据

    # croze 用于衡量 net 和 advnet 之间的相似性
    def croze(layer, layer_adv, feat, feat_adv):
        if layer.weight.grad is not None:                               # 计算两个模型每一层的权重值
            w_sim = (1+cos_loss(layer_adv.weight, layer.weight)).sum()  # 权重相似性
            sim = (torch.abs(cos_loss(layer_adv.weight.grad, layer.weight.grad))).sum()  # 梯度相似性, 有些情况下梯度相似性是接近于0的
            feat_sim = (1+cos_loss(feat_adv, feat)).sum()               # 每层的特征相似性
            # print(torch.abs(w_sim * sim * feat_sim))
            # print(f"w_sim: {w_sim.item():.3e}, sim: {sim.item():.3e}, feat_sim: {feat_sim.item():.3e} \t")

            return torch.abs(w_sim * sim * feat_sim)
            # return torch.abs(sim * feat_sim)          # 梯度相似性与特征相似性相乘RB201    sp0.785  DARTS sp 0.41
            # return torch.abs(w_sim * sim)             # 权重与梯度相似性相乘RB201         sp0.81   DARTS sp 0.44
            # return torch.abs(w_sim * feat_sim)        # 权重与特征相似性相乘RB201         sp0.62   DARTS sp 0.51
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array_adv_feats(net, advnet, feats, adv_feats, croze, mode, search_space)

    # apply signs of all params
    # 把参数的符号都返回去
    nonlinearize(net, signs)
    nonlinearize(advnet, adv_signs)
    
    del feats, adv_feats
    del advnet

    # print(len(grads_abs))

    return grads_abs
