"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for creating the dataloader for training/validation/test phase
"""

import sys

import torch
from torch.utils.data import DataLoader
import numpy as np

sys.path.append('../')

from data_process.kitti_dataset import KittiDataset
from data_process.transformation import OneOf, Random_Rotation, Random_Scaling, Random_Rotate_Individual_Box


def create_train_dataloader(configs):
    """Create dataloader for training"""
    train_lidar_aug = OneOf([
        Random_Rotation(limit_angle=np.pi / 4, p=1.0),
        Random_Scaling(scaling_range=(0.95, 1.05), p=1.0),
        Random_Rotate_Individual_Box(limit_angle=np.pi / 10, p=1.0)
    ], p=0.75)
    train_dataset = KittiDataset(configs, mode='train', lidar_aug=train_lidar_aug, aug_transforms=None,
                                 num_samples=configs.num_samples)
    train_sampler = None
    if configs.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=(train_sampler is None),
                                  pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=train_sampler,
                                  collate_fn=train_dataset.collate_fn)

    return train_dataloader, train_sampler


def create_val_dataloader(configs):
    """Create dataloader for validation"""
    val_sampler = None
    val_dataset = KittiDataset(configs, mode='val', lidar_aug=None, aug_transforms=None,
                               num_samples=configs.num_samples)
    if configs.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                                pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=val_sampler,
                                collate_fn=val_dataset.collate_fn)

    return val_dataloader


def create_test_dataloader(configs):
    """Create dataloader for testing phase"""

    test_dataset = KittiDataset(configs, mode='test', lidar_aug=None, aug_transforms=None,
                                num_samples=configs.num_samples)
    test_sampler = None
    if configs.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False,
                                 pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=test_sampler,
                                 collate_fn=test_dataset.collate_fn_test)

    return test_dataloader
