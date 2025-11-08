import os.path as osp
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset


class RandChannel(object):
    # randomly pick channels from input
    def __init__(self, num_channel):
        self.num_channel = num_channel

    def __repr__(self):
        return ('{name}(num_channel={num_channel})'.format(name=self.__class__.__name__, **self.__dict__))

    def __call__(self, img):
        channel = img.size(0)
        channel_choice = sorted(np.random.choice(list(range(channel)), size=self.num_channel, replace=False))
        return torch.index_select(img, 0, torch.Tensor(channel_choice).long())


def get_c10(name, root, input_size, cutout=-1):
    assert len(input_size) in [3, 4]
    if len(input_size) == 4:
        input_size = input_size[1:]
    assert input_size[1] == input_size[2]

    if name == 'cifar10':
        mean = [0.49139968, 0.48215827, 0.44653124]
        std  = [0.24703233, 0.24348505, 0.26158768]
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    # Data Argumentation
    if name == 'cifar10' or name == 'cifar100':
        lists = [transforms.RandomCrop(input_size[1], padding=0), transforms.ToTensor(), transforms.Normalize(mean, std), RandChannel(input_size[0])]
        train_transform = transforms.Compose(lists)
        test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    if name == 'cifar10':
        train_data = dset.CIFAR10(root, train=True , transform=train_transform, download=True)
        test_data  = dset.CIFAR10(root, train=False, transform=test_transform , download=True)
        assert len(train_data) == 50000 and len(test_data) == 10000
    else: raise TypeError("Unknow dataset : {:}".format(name))

    class_num = 10
    # print(train_data)
    return train_data, test_data, class_num
