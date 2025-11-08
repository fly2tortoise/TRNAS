# # Copyright 2021 Samsung Electronics Co., Ltd.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
#
# #     http://www.apache.org/licenses/LICENSE-2.0
#
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.


# from . import measures
import torch
import torch.nn.functional as F
from .p_utils import *
from . import measures

import types
import copy

def no_op(self,x):
    return x


def copynet(self, bn):
    net = copy.deepcopy(self)
    if bn==False:
        for l in net.modules():
            if isinstance(l,nn.BatchNorm2d) or isinstance(l,nn.BatchNorm1d) :
                l.forward = types.MethodType(no_op, l)
    return net

def trnas_c_no(net_orig, dataloader, attacker=None, loss_fn=F.cross_entropy):
    def sum_arr(arr):
        # 累加 arr
        sum = 0.
        for i in range(len(arr)): 
            sum += torch.sum(arr[i])  #
        if isinstance(sum, float):
            return sum
        return sum.item()

    measure_names = ['conjs']
    inputs, targets = get_some_data(dataloader, num_batches=1, device="cuda")
    measure_values = {}

    if attacker is not None:
        inputs = attacker.attack(inputs, targets)  # 攻击模型生成的 inputs
        attacker.model.zero_grad()

    # 查看网络架构中，是否有 get_prunable_copy 这个选项，用于新建model，避免bn中参数改变
    if not hasattr(net_orig, 'get_prunable_copy'):
        net_orig.get_prunable_copy = types.MethodType(copynet, net_orig)

    # move to cpu to free up mem
    torch.cuda.empty_cache()
    net_orig = net_orig.cpu()
    torch.cuda.empty_cache()

    # 简化版：直接计算所有 measure
    from . import measures
    for measure_name in measure_names:
        val = measures.calc_measure(measure_name, net_orig, "cuda", inputs, targets, loss_fn=loss_fn, split_data=1, search_space='darts')
        measure_values[measure_name] = val
    net_orig = net_orig.to("cuda").train()
    measures_arr = measure_values

    measures = {}
    for k,v in measures_arr.items():
        # print(k, len(v)) # 这里的 k代表的是代理类型, v只有1条list
        if v == None:
            return None
        else:
            measures[k] = sum_arr(v)
    _, acc_croze = list(measures.items())[0]

    return acc_croze

