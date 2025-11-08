# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

available_measures = []
_measure_impls = {}

# 调用measure装饰器，动态注册函数
def measure(name, bn=True, copy_net=True, force_clean=True, **impl_args):
    def make_impl(func):
        # 复制网络或直接使用原网络
        def measure_impl(net_orig, device, *args, **kwargs): # _measure_impls 是一个全局字典
            if copy_net: # 一般是这个
                net = net_orig.get_prunable_copy(bn=bn).to(device)
            else:
                net = net_orig
            # 执行实际的测量函数（func 是传入的测量函数）
            ret = func(net, *args, **kwargs, **impl_args)
            # 清理内存
            if copy_net and force_clean:
                import gc
                import torch
                del net
                torch.cuda.empty_cache()
                gc.collect()
            return ret

        # 注册测量函数到全局字典
        global _measure_impls
        if name in _measure_impls:
            raise KeyError(f'Duplicated measure! {name}')
        available_measures.append(name)
        _measure_impls[name] = measure_impl
        return func

    return make_impl  #  返回原始函数


def calc_measure(name, net, device, *args, **kwargs):
    # 根据已注册好的评价函数，执行评价
    # input: measure_name, net_orig, device, inputs, targets, loss_fn=loss_fn, split_data=ds, search_space=search_space
    # print(name, net, device)  # croze, net, cuda:0          # 鲁棒输入 inputs, targets
    # print(_measure_impls)
    return _measure_impls[name](net, device, *args, **kwargs) # 相当于这里是 croze(net,device, xxx,xxx)

def load_all():
    # from . import croze
    from . import conjs

load_all()
