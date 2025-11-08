from __future__ import print_function
import os
import argparse
import sys
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from model import NetworkCIFAR as Network
import genotypes
from tqdm import tqdm
import pandas as pd
from datetime import datetime

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',help='input batch size for testing (default: 200)')
parser.add_argument('--base_dir', type=str, default='../train/eval-EXP-20250917-214347-xxx', help='path to store log file')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--epsilon', default=8/255, help='Total pertubation scale, 8 / 255 in this paper.') # 8/255
parser.add_argument('--num-steps', type=int, default=20, help='perturb number of steps')                # PGD 20 100
parser.add_argument('--step-size', default=2/255,help='step-size, 8 / 255 for FGSM, and 2 / 255 for PGD')
parser.add_argument('--random',default=True, help='True for PGD and False for FGSM')
parser.add_argument('--arch', type=str, default='TRNAS', help='which architecture to use')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--data_type', type=str, default='cifar10', help='which dataset to use')
parser.add_argument('--iteration', type=int, default=101, help='Iteration in test')
parser.add_argument('--max_epoch', type=int, default=120, help='Stop in PGD test')
parser.add_argument('--checkpoint_epoch', type=int, default=0, help='Iteration in test')
args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# set up data loader
if args.data_type == 'cifar10':
    transform_list = [transforms.ToTensor()]
    transform_test = transforms.Compose(transform_list)
    testset = torchvision.datasets.CIFAR10(root='../../0_Data', train=False, download=True, transform=transform_test)

elif args.data_type == 'cifar100':
    transform_list = [transforms.ToTensor()]
    transform_test = transforms.Compose(transform_list)
    testset = torchvision.datasets.CIFAR100(root='../../0_Data', train=False, download=True, transform=transform_test)


test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def _pgd_whitebox(model, X, y, epsilon=args.epsilon, num_steps=args.num_steps, step_size=args.step_size):

    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()

    # print('err pgd (white-box): ', err_pgd) # 打印对抗样本的分类错误数量。

    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    # 初始化计数器
    counter = 0
    is_tty = sys.stdout.isatty()
    for data, target in tqdm(test_loader, desc="Testing Progress", disable=not is_tty): # # for data, target in tqdm(test_loader, desc="Testing", disable=not is_tty):
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        # print(X.size()) # torch.Size([100, 3, 32, 32])
        err_natural, err_robust = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
        # break
        counter += 1
        # if counter % args.iteration == 0:
        #     print(f"Iteration {counter}: Natural error {err_natural}, Robust error {err_robust}")


    acc = 100 - (natural_err_total/100).item()
    rob = 100 - (robust_err_total /100).item()

    return acc, rob


def main():


    print('pgd white-box attack')
    if args.data_type == 'cifar100':
        CIFAR_CLASSES = 100
    else:
        CIFAR_CLASSES = 10
    genotype = eval("genotypes.%s" % args.arch)

    # 测试当前文件夹内所有weights的 PGD accs
    base_dir = args.base_dir
    # 获取所有检查点文件
    checkpoint_files = [f for f in os.listdir(base_dir) if f.startswith('checkpoint-epoch') and f.endswith('.pth.tar')]
    # 提取 epoch 编号并排序
    checkpoints = []
    for file in checkpoint_files:
        match = re.search(r'epoch(\d+)', file)
        if match:
            epoch = int(match.group(1))
            checkpoints.append((epoch, file))
    checkpoints.sort()  # 按 epoch 升序排序

    # 初始化数据列表
    epochs = []
    cleans = []
    pgds   = []
    best_pgd = float('-inf')  # 跟踪当前最优 PGD
    best_clean = float('-inf')  # 跟踪当前最优 Clean
    # 遍历所有检查点文件
    for epoch, file_path in checkpoints:

        dir_path = os.path.join(base_dir, file_path)
        if not os.path.exists(dir_path):
            print(f"Checkpoint {dir_path} does not exist, skipping...")
            continue
        if epoch > args.max_epoch:
            print(f"Reached max epoch {args.max_epoch}, stopping...")
            break

        # 初始化模型
        model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
        # 加载检查点
        checkpoint = torch.load(dir_path, map_location='cuda:0')
        model.load_state_dict(checkpoint['state_dict'])
        # 配置模型
        model.drop_path_prob = args.drop_path_prob
        model.cuda()
        # 评估性能
        clean_acc, pgd_acc = eval_adv_test_whitebox(model, device, test_loader)
        # 打印结果
        print(f"Epoch {epoch}: Clean = {clean_acc:.4f}, PGD = {pgd_acc:.4f}")

        # 更新最优 clean 和 PGD
        if clean_acc > best_clean:
            best_clean = clean_acc
        if pgd_acc > best_pgd:
            best_pgd = pgd_acc
        # 记录数据
        epochs.append(epoch)
        cleans.append(clean_acc)
        pgds.append(pgd_acc)

    # 使用 pandas 创建转置 DataFrame 并保存为 CSV
    data = {'Epoch': epochs, 'Clean': cleans, 'PGD': pgds}
    df = pd.DataFrame(data)
    df_transposed = df.set_index('Epoch').T

    # 添加 Best 列（最后一列）
    df_transposed['Best'] = [best_clean, best_pgd]

    # 获取当前时间和架构信息
    current_time = datetime.now().strftime('%m-%d_%H-%M')
    # 构造输出文件名，包含时间和架构
    output_file = f'0.pgd_{current_time}_{args.arch}.xlsx'
    # 保存到 Excel
    df_transposed.to_excel(output_file)
    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    main()
