
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
import os
import sys
import hashlib
from PIL import Image
import torch
import torchvision
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms
if sys.version_info[0] == 2:
  import cPickle as pickle
else:
  import pickle
  
from .corrupt_dataset import *
from .sampler import *


def get_num_classes(args):
    return 100 if args.dataset == 'cifar100' else 10 if args.dataset == 'cifar10' else 120


def get_dataloaders(
    train_batch_size, 
    test_batch_size, 
    dataset, 
    num_workers, 
    resize=None, 
    datadir='_dataset', 
    normalize=True,
    return_ds=False,
    corruption=None,
    severity=None
):


    if 'cifar' in dataset:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        size, pad = 32, 4

    if resize is None:
        resize = size

    train_transform_list = [
        transforms.RandomCrop(size, padding=pad),
        transforms.Resize(resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    test_transform_list = [
        transforms.Resize(resize),
        transforms.ToTensor(),
    ]
    
    if normalize:
        train_transform_list.append(transforms.Normalize(mean,std))
        test_transform_list.append(transforms.Normalize(mean,std))

    train_transform = transforms.Compose(train_transform_list)
    test_transform = transforms.Compose(test_transform_list)
        
    if dataset == 'cifar10':
        train_dataset = CIFAR10(datadir, True, train_transform, download=True)
        test_dataset = CIFAR10(datadir, False, test_transform, download=True)

    else:
        raise ValueError('There are no more cifars or imagenets.')


    train_loader = DataLoader(
        train_dataset,
        train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)
    test_loader = DataLoader(
        test_dataset,
        test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    return train_loader, test_loader


##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################

def calculate_md5(fpath, chunk_size=1024 * 1024):
  md5 = hashlib.md5()
  with open(fpath, 'rb') as f:
    for chunk in iter(lambda: f.read(chunk_size), b''):
      md5.update(chunk)
  return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
  return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
  if not os.path.isfile(fpath): return False
  if md5 is None: return True
  else          : return check_md5(fpath, md5)




  