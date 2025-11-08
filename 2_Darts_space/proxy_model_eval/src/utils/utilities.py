import numpy as np
import torch
import torch.nn as nn

class Model(object):
    def __init__(self):
        self.arch = None
        self.geno = None
        self.score = None

def count_parameters(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e3


def network_weight_gaussian_init(net: nn.Module): # 高斯初始化函数
    with torch.no_grad():
        for m in net.modules():
            # print(m)
            # 1. 初始化 Conv2d 卷积层权重为正态分布, 初始化bias 卷积层偏置为0
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            # 2. 初始化批归一化/组归一化层的权重为1, bias为0
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            # 3. 初始化线性层（全连接层）权重为正态分布, bias为0
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue
    return net

def network_weight_gaussian_init_201(net: nn.Module): # 高斯初始化函数
    with torch.no_grad():
        for m in net.modules():
            # print(f"Init: {m.__class__.__name__}")


            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

            # 1. 初始化 Conv2d 卷积层权重为正态分布, 初始化bias 卷积层偏置为0
            # if isinstance(m, nn.Conv2d):
            #     nn.init.normal_(m.weight)
            #     if hasattr(m, 'bias') and m.bias is not None:
            #         nn.init.zeros_(m.bias)
            # # 2. 初始化批归一化/组归一化层的权重为1, bias为0
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            #     nn.init.ones_(m.weight)
            #     nn.init.zeros_(m.bias)
            # # 3. 初始化线性层（全连接层）权重为正态分布, bias为0
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight)
            #     if hasattr(m, 'bias') and m.bias is not None:
            #         nn.init.zeros_(m.bias)
            # else:
            #     continue
    return net

def network_weight_gaussian_init_201_(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            # print(f"Init: {m.__class__.__name__}")
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)  # 只初始化非空的 weight
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # 只初始化非空的 bias

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    return net


if __name__ == '__main__':
    pass


