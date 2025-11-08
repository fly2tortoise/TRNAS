import sys

from .operations import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

def drop_path(x, drop_prob):
  # 随机路径丢弃
  if drop_prob > 0.:
    x = nn.functional.dropout(x, p=drop_prob)

  return x

class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()

    # preprocess 主要是对前面 4条聚合的通道数进行拆分。 拆分后的通道数 s0 和 s1 是相同的
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)  # 需要面对通道数 x 2
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, 1, True)

    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, 1, True)

    # 读取 concat  节点 [2, 3, 4, 5]
    if reduction:
        op_names, indices = zip(*genotype.reduce)  # indices:  (0, 0, 0, 1, 1, 2, 3, 4) 每个模型是不同的
        # print("reduction",op_names, indices)
        # normal ('none', 'dil_conv_5x5', 'sep_conv_5x5', 'sep_conv_3x3', 'sep_conv_3x3', 'dil_conv_5x5', 'sep_conv_5x5', 'skip_connect')
        concat = genotype.reduce_concat #  [2, 3, 4, 5]
    else:
        op_names, indices = zip(*genotype.normal)
        # print("normal",op_names, indices)
        concat = genotype.normal_concat #  [2, 3, 4, 5]

    self._compile(C, op_names, indices, concat, reduction) # C, (operations), (indices), [2, 3, 4, 5], xx

  def _compile(self, C, op_names, indices, concat, reduction):
      # 主要目的: 组装操作集合 _ops,
    assert len(op_names) == len(indices)  # 8,  8

    self._steps = len(op_names) // 2    # 4
    self._concat = concat               # [2, 3, 4, 5]
    self.multiplier = len(concat)       # 4
    self._ops = nn.ModuleList()

    for name, index in zip(op_names, indices):       # len 都是 8
        stride = 2 if reduction and index < 2 else 1 # 图像的降维由cell内的操作实际执行  通道数变化在s0 s1输入时就已经更新
        op = OPS[name](C, C, stride, True)
        self._ops += [op]
        # print("name, index, type(op):", name, index, type(op).__name__) # 不同结构都不一样
    self._indices = indices

  def forward(self, s0, s1, drop_prob):

    # print("befor_s0_s1", s0.size(), s1.size())
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    # print('after_s0_s1', s0.size(), s1.size())  # reduction 情况下，s0未降采样，s1已降采样

    # cell 核心
    states = [s0, s1]
    # print(self._indices)
    for i in range(self._steps):            # 4
      # print("edge idx:", 2*i, 2*i+1)      # 按照Genotype里的顺序进行分配
      # print(self._indices[2*i], self._indices[2*i+1]) # Genotype里op 对应的idx
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      # print(h1.size(), h2.size())
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      # print(self._ops[2*i], self._ops[2*i+1]) # 这里是对应的操作名
      h1 = op1(h1)
      h2 = op2(h2)
      # 是否丢弃路径,
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]

    return torch.cat([states[i] for i in self._concat], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, genotype):
        self.drop_path_prob = 0.
        super(Network, self).__init__()

        self._layers = layers

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = False

        # 本文layer计算的是 1
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:  # 通道数x2, 特征图大小减半。 层数小于3层，都是reduction
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            # print("C_prev_prev, C_prev, C_curr:", C_prev_prev, C_prev, C_curr)
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            # print(cell)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input, return_feats_only=False, return_activated_feats=False, consist=True):
        s0 = s1 = input
        activated_feats = []
        if consist:
            activated_feats.append(s1.clone())
        # print(self.cells) # 令人奇怪的是只有一个cell？  的确如此，只有1个layer
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)  # s1先算一步
            # print(i, s0.size(), s1.size(), "\n")
            if consist:
                activated_feats.append(s1.clone())

        out = self.global_pooling(s1)
        if consist:
            activated_feats.append(out.clone())

        if return_feats_only and consist:
            return activated_feats

        out = out.view(out.size(0), -1)
        # print(out.size())
        logits = self.classifier(out)  # 把通道数压缩为 10 (batchsize, 10)

        if return_activated_feats and consist:
            return logits, activated_feats

        del activated_feats

        return logits

class Network_c(nn.Module):

    def __init__(self, C, num_classes, layers, genotype):
        self.drop_path_prob = 0.
        super(Network_c, self).__init__()

        self._layers = layers

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = False

        # 本文layer计算的是 1
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:  # 通道数x2, 特征图大小减半。 层数小于3层，都是reduction
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            # print("C_prev_prev, C_prev, C_curr:", C_prev_prev, C_prev, C_curr)
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            # print(cell)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input, return_feats_only=False, return_activated_feats=False, consist=True):
        s0 = s1 = input
        activated_feats = []
        if consist:
            activated_feats.append(s1.clone())
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)  # s1先算一步
            if consist:
                activated_feats.append(s1.clone())

        out = self.global_pooling(s1)

        if consist:
            activated_feats.append(out.clone())
        if return_feats_only and consist:
            return activated_feats

        out = out.view(out.size(0), -1)
        # print(out.size())
        logits = self.classifier(out)  # 把通道数压缩为 10 (batchsize, 10)

        if return_activated_feats and consist:
            return logits, activated_feats

        del activated_feats

        return logits


class Network_l(nn.Module):

    def __init__(self, C, num_classes, layers, genotype):
        self.drop_path_prob = 0.
        super(Network_l, self).__init__()

        self._layers = layers

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = False

        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            # print("C_prev_prev, C_prev, C_curr:", C_prev_prev, C_prev, C_curr)
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            # print(cell)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input, return_feats_only=False, return_activated_feats=False, consist=True):
        s0 = s1 = input
        # activated_feats = []
        # if consist:
        #     activated_feats.append(s1.clone())
        # print(self.cells) # 令人奇怪的是只有一个cell？  的确如此，只有1个layer
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)  # s1先算一步
            # print(i, s0.size(), s1.size(), "\n")
            # if consist:
            #     activated_feats.append(s1.clone())

        out = self.global_pooling(s1)
        # if consist:
        #     activated_feats.append(out.clone())

        # if return_feats_only and consist:
        #     return activated_feats

        out = out.view(out.size(0), -1)
        # print(out.size())
        logits = self.classifier(out)  # 把通道数压缩为 10 (batchsize, 10)

        # if return_activated_feats and consist:
        #     return logits, activated_feats
        #
        # del activated_feats

        return logits

