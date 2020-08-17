"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: The centerNet3D model
"""

import sys

import torch.nn as nn
import spconv

sys.path.append('../')

from models.deform_conv_v2 import DeformConv2d


class CenterNet3D(nn.Module):
    def __init__(self, sparse_shape, heads, head_conv, num_input_features=4):
        super(CenterNet3D, self).__init__()

        self.sparse_shape = sparse_shape
        self.heads = heads

        self.net3d = spconv.SparseSequential(
            spconv.SubMConv3d(num_input_features, 16, 3, 1, 1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            spconv.SubMConv3d(16, 16, 3, 1, 1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            spconv.SparseConv3d(16, 32, 3, 2, 1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),

            spconv.SubMConv3d(32, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            spconv.SubMConv3d(32, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            spconv.SparseConv3d(32, 64, 3, 2, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),

            spconv.SubMConv3d(64, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            spconv.SubMConv3d(64, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            spconv.SubMConv3d(64, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            spconv.SparseConv3d(64, 64, 3, 2, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            spconv.SubMConv3d(64, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            spconv.SubMConv3d(64, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            spconv.SubMConv3d(64, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True)
        )

        self.net2d = nn.Sequential(
            nn.Conv2d(320, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            DeformConv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 3, 2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )

        for head in sorted(self.heads):
            num_output = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(128, head_conv, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_output, kernel_size=1, stride=1, padding=0))

            self.__setattr__('head_{}'.format(head), fc)

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        feat_voxelization = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        feature_3d = self.net3d(feat_voxelization)
        feature_3d = feature_3d.dense()
        N, C, D, H, W = feature_3d.shape
        feature_3d = feature_3d.view(N, C * D, H, W)
        feature_2d = self.net2d(feature_3d)

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__('head_{}'.format(head))(feature_2d)

        return ret


if __name__ == '__main__':
    import torch
    from easydict import EasyDict as edict

    configs = edict()

    configs.num_classes = 3
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos
    configs.num_conners = 4

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cenoff': configs.num_center_offset,
        'direction': configs.num_direction,
        'z': configs.num_z,
        'dim': configs.num_dim,
        'hm_conners': configs.num_classes  # equal classes --> 3
    }

    configs.head_conv = 64
    configs.sparse_shape = (40, 1600, 1400)

    model = CenterNet3D(configs.sparse_shape, configs.heads, configs.head_conv, num_input_features=4)
    print(model)
