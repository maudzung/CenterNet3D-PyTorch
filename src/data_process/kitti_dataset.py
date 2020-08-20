"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for the KITTI dataset
# Refer from: https://github.com/xingyizhou/CenterNet
"""

import sys
import os
import math
from builtins import int

import numpy as np
from torch.utils.data import Dataset
import cv2
import spconv
import torch
import mayavi.mlab as mlab

sys.path.append('../')

from data_process.kitti_data_utils import gen_hm_radius, compute_radius, Calibration, get_filtered_lidar, \
    box3d_corners_to_center
from data_process import transformation
from utils.visualization_utils import draw_lidar, draw_gt_boxes3d
import config.kitti_config as cnf


class KittiDataset(Dataset):
    def __init__(self, configs, mode='train', lidar_aug=None, aug_transforms=None, num_samples=None):
        self.dataset_dir = configs.dataset_dir
        self.input_size = configs.input_size
        self.hm_size = configs.hm_size
        self.down_ratio = configs.down_ratio

        self.num_classes = configs.num_classes
        self.max_objects = configs.max_objects
        self.num_conners = configs.num_conners
        self.num_input_features = configs.num_input_features

        assert mode in ['train', 'val', 'test'], 'Invalid mode: {}'.format(mode)
        self.mode = mode
        self.is_test = (self.mode == 'test')
        sub_folder = 'testing' if self.is_test else 'training'

        self.lidar_aug = lidar_aug
        self.aug_transforms = aug_transforms

        self.image_dir = os.path.join(self.dataset_dir, sub_folder, "image_2")
        self.lidar_dir = os.path.join(self.dataset_dir, sub_folder, "velodyne")
        self.calib_dir = os.path.join(self.dataset_dir, sub_folder, "calib")
        self.label_dir = os.path.join(self.dataset_dir, sub_folder, "label_2")
        split_txt_path = os.path.join(self.dataset_dir, 'ImageSets', '{}.txt'.format(mode))
        self.sample_id_list = [int(x.strip()) for x in open(split_txt_path).readlines()]

        if num_samples is not None:
            self.sample_id_list = self.sample_id_list[:num_samples]
        self.num_samples = len(self.sample_id_list)

        self.voxel_generator = spconv.utils.VoxelGenerator(
            voxel_size=[0.05, 0.05, 0.1],
            point_cloud_range=[0, -40, -3, 70, 40, 1],
            max_num_points=30,
            max_voxels=40000,
            full_mean=False
        )

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, index):
        if self.is_test:
            return self.load_img_only(index)
        else:
            return self.load_img_with_targets(index)

    def load_img_only(self, index):
        """Load only image for the testing phase"""

        sample_id = int(self.sample_id_list[index])
        lidarData = self.get_lidar(sample_id)
        voxels_features, voxels_coors, num_points_per_voxel = self.voxel_generator.generate(lidarData)

        voxels_features = torch.from_numpy(voxels_features)
        voxels_coors = torch.from_numpy(voxels_coors)
        num_points_per_voxel = torch.from_numpy(num_points_per_voxel)

        voxels_features = self.simplified_voxel(voxels_features, num_points_per_voxel,
                                                self.num_input_features)

        return sample_id, lidarData, voxels_features, voxels_coors

    def load_img_with_targets(self, index):
        """Load images and targets for the training and validation phase"""
        sample_id = int(self.sample_id_list[index])
        lidarData = self.get_lidar(sample_id)
        calib = self.get_calib(sample_id)
        labels, has_labels = self.get_label(sample_id)
        gt_boxes3d = self.get_gt_boxes3d_cam2velo(labels, has_labels, calib)

        if self.lidar_aug:
            lidarData, gt_boxes3d = self.lidar_aug(lidarData, gt_boxes3d)

        lidarData, gt_boxes3d, cls_ids = get_filtered_lidar(lidarData, gt_boxes3d, labels[:, 0])

        gt_hwlxyzr = box3d_corners_to_center(gt_boxes3d)
        targets = self.build_targets(cls_ids, gt_hwlxyzr, gt_boxes3d)
        # targets = None
        voxels_features, voxels_coors, num_points_per_voxel = self.voxel_generator.generate(lidarData)

        voxels_features = torch.from_numpy(voxels_features)
        voxels_coors = torch.from_numpy(voxels_coors)
        num_points_per_voxel = torch.from_numpy(num_points_per_voxel)

        voxels_features = self.simplified_voxel(voxels_features, num_points_per_voxel,
                                                self.num_input_features)

        # return lidarData, gt_boxes3d, cls_ids, voxels_features, voxels_coors, targets
        return voxels_features, voxels_coors, targets

    def simplified_voxel(self, voxels_features, num_points_per_voxel, num_input_features):
        points_sum = voxels_features[:, :, :num_input_features].sum(dim=1, keepdim=False)
        points_mean = points_sum / num_points_per_voxel.type_as(voxels_features).view(-1, 1)

        return points_mean.contiguous()

    def collate_fn(self, batchdata):
        b_voxel_features = []
        b_voxel_coords = []
        b_targets = {}
        batch_size = len(batchdata)
        for i, sample_data in enumerate(batchdata):
            features, coors, targets = sample_data
            b_voxel_features.append(features)
            # b_voxel_coords.append(F.pad(coors, ((0, 0), (1, 0)), mode='constant', value=i))
            b_voxel_coords.append(np.pad(coors, ((0, 0), (1, 0)), mode='constant', constant_values=i))
            for key, val in targets.items():
                if i == 0:
                    b_targets[key] = [val]
                else:
                    b_targets[key].append(val)

        for key, val in b_targets.items():
            b_targets[key] = torch.from_numpy(np.stack(val))

        return batch_size, torch.cat(b_voxel_features), torch.from_numpy(np.concatenate(b_voxel_coords)), b_targets

    def collate_fn_test(self, batchdata):
        b_voxel_features = []
        b_voxel_coords = []
        b_sample_id = []
        b_lidarData = []
        batch_size = len(batchdata)
        for i, sample_data in enumerate(batchdata):
            sample_id, lidarData, features, coors = sample_data
            b_sample_id.append(sample_id)
            b_lidarData.append(lidarData)
            b_voxel_features.append(features)
            b_voxel_coords.append(np.pad(coors, ((0, 0), (1, 0)), mode='constant', constant_values=i))

        return b_sample_id, batch_size, b_lidarData, torch.cat(b_voxel_features), \
               torch.from_numpy(np.concatenate(b_voxel_coords))

    def get_image(self, idx):
        img_path = os.path.join(self.image_dir, '{:06d}.png'.format(idx))
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        return img_path, img

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '{:06d}.txt'.format(idx))
        # assert os.path.isfile(calib_file)
        return Calibration(calib_file)

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '{:06d}.bin'.format(idx))
        # assert os.path.isfile(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_label(self, idx):
        labels = []
        label_path = os.path.join(self.label_dir, '{:06d}.txt'.format(idx))
        for line in open(label_path, 'r'):
            line = line.rstrip()
            line_parts = line.split(' ')
            obj_name = line_parts[0]  # 'Car', 'Pedestrian', ...
            cat_id = int(cnf.CLASS_NAME_TO_ID[obj_name])
            if cat_id <= -99:  # ignore Tram and Misc
                continue
            truncated = int(float(line_parts[1]))  # truncated pixel ratio [0..1]
            occluded = int(line_parts[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
            alpha = float(line_parts[3])  # object observation angle [-pi..pi]
            # xmin, ymin, xmax, ymax
            bbox = np.array([float(line_parts[4]), float(line_parts[5]), float(line_parts[6]), float(line_parts[7])])
            # height, width, length (h, w, l)
            h, w, l = float(line_parts[8]), float(line_parts[9]), float(line_parts[10])
            # location (x,y,z) in camera coord.
            x, y, z = float(line_parts[11]), float(line_parts[12]), float(line_parts[13])
            ry = float(line_parts[14])  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

            object_label = [cat_id, h, w, l, x, y, z, ry]
            labels.append(object_label)

        if len(labels) == 0:
            labels = np.zeros((1, 8), dtype=np.float32)
            has_labels = False
        else:
            labels = np.array(labels, dtype=np.float32)
            has_labels = True

        return labels, has_labels

    def build_targets(self, cls_ids, hwlxyzr, boxes3d):
        minX = cnf.boundary['minX']
        maxX = cnf.boundary['maxX']
        minY = cnf.boundary['minY']
        maxY = cnf.boundary['maxY']
        minZ = cnf.boundary['minZ']
        maxZ = cnf.boundary['maxZ']

        num_objects = min(len(cls_ids), self.max_objects)
        hm_l, hm_w = self.hm_size

        hm_main_center = np.zeros((self.num_classes, hm_l, hm_w), dtype=np.float32)
        cen_offset = np.zeros((self.max_objects, 2), dtype=np.float32)
        direction = np.zeros((self.max_objects, 2), dtype=np.int64)
        z_coor = np.zeros((self.max_objects, 1), dtype=np.float32)
        dimension = np.zeros((self.max_objects, 3), dtype=np.float32)
        hm_conners = np.zeros((self.num_classes, hm_l, hm_w), dtype=np.float32)

        indices_center = np.zeros((self.max_objects), dtype=np.int64)
        obj_mask = np.zeros((self.max_objects), dtype=np.uint8)

        for k in range(num_objects):
            cls_id = int(cls_ids[k])
            h, w, l, x, y, z, yaw = hwlxyzr[k]
            if not ((minX <= x <= maxX) and (minY <= y <= maxY) and (minZ <= z <= maxZ)):
                continue
            if (h <= 0) or (w <= 0) or (l <= 0):
                continue
            box_3d = boxes3d[k]
            bbox_l = l / cnf.bound_size_x * hm_l
            bbox_w = w / cnf.bound_size_y * hm_w
            radius = compute_radius((math.ceil(bbox_l), math.ceil(bbox_w)))
            radius = max(0, int(radius))

            center_y = (x - minX) / cnf.bound_size_x * hm_l  # x --> y (invert to 2D image space)
            center_x = (y - minY) / cnf.bound_size_y * hm_w  # y --> x
            center = np.array([center_x, center_y], dtype=np.float32)
            center_int = center.astype(np.int32)
            if cls_id < 0:
                ignore_ids = [_ for _ in range(self.num_classes)] if cls_id == - 1 else [- cls_id - 2]
                # Consider to make mask ignore
                for cls_ig in ignore_ids:
                    gen_hm_radius(hm_main_center[cls_ig], center_int, radius)
                hm_main_center[ignore_ids, center_int[1], center_int[0]] = 0.9999
                continue

            # Generate heatmaps for main center
            gen_hm_radius(hm_main_center[cls_id], center, radius)
            # Index of the center
            indices_center[k] = center_int[1] * hm_w + center_int[0]

            bev_conners = box_3d[:4, :2].copy()  # just take x,y (unit: meter)
            bev_conners[:, 0] = (bev_conners[:, 0] - minX) / cnf.bound_size_x * hm_l
            bev_conners[:, 1] = (bev_conners[:, 1] - minY) / cnf.bound_size_y * hm_w
            bev_conners = bev_conners[:, [1, 0]]  # x --> y, y --> x

            for conner_idx, conner in enumerate(bev_conners):
                conner_int = conner.astype(np.int32)
                if (0 <= conner_int[0] < hm_w) and (0 <= conner_int[1] < hm_l):
                    # targets for conners
                    gen_hm_radius(hm_conners[cls_id], conner, radius)  # cls_id instead of conner_idx

            # targets for center offset
            cen_offset[k] = center - center_int

            # targets for dimension
            dimension[k, 0] = h
            dimension[k, 1] = w
            dimension[k, 2] = l

            # targets for direction
            direction[k, 0] = np.sin(yaw)
            direction[k, 1] = np.cos(yaw)

            # targets for depth
            z_coor[k] = z

            # Generate object masks
            obj_mask[k] = 1

        targets = {
            'hm_cen': hm_main_center,
            'cen_offset': cen_offset,
            'direction': direction,
            'z_coor': z_coor,
            'dim': dimension,
            'hm_conners': hm_conners,
            'indices_center': indices_center,
            'obj_mask': obj_mask,
        }

        return targets

    def draw_img_with_label(self, index):
        sample_id = int(self.sample_id_list[index])
        lidarData = self.get_lidar(sample_id)
        calib = self.get_calib(sample_id)
        labels, has_labels = self.get_label(sample_id)
        gt_boxes3d = self.get_gt_boxes3d_cam2velo(labels, has_labels, calib)

        if self.lidar_aug:
            lidarData, gt_boxes3d = self.lidar_aug(lidarData, gt_boxes3d)

        lidarData, gt_boxes3d, cls_ids = get_filtered_lidar(lidarData, gt_boxes3d, labels[:, 0])

        return lidarData, gt_boxes3d, cls_ids

    def get_gt_boxes3d_cam2velo(self, labels, has_labels, calib):
        if has_labels:
            gt_boxes3d = []
            for label in labels:
                box3d_corner = transformation.box3d_cam_to_velo(label[1:], calib.V2C)
                gt_boxes3d.append(box3d_corner)
            gt_boxes3d = np.array(gt_boxes3d).reshape(-1, 8, 3)
        else:
            gt_boxes3d = np.zeros(shape=(1, 8, 3))

        return gt_boxes3d


if __name__ == '__main__':
    from easydict import EasyDict as edict
    from data_process.transformation import OneOf, Random_Scaling, Random_Rotation, Random_Rotate_Individual_Box

    configs = edict()
    configs.distributed = False  # For testing
    configs.pin_memory = False
    configs.num_samples = None
    configs.input_size = (1400, 1600)
    configs.hm_size = (350, 400)
    configs.down_ratio = 4
    configs.max_objects = 50
    configs.num_classes = 3
    configs.num_conners = 4
    configs.num_input_features = 4

    configs.dataset_dir = os.path.join('../../', 'dataset', 'kitti')
    lidar_aug = OneOf([
        Random_Rotation(limit_angle=np.pi / 4, p=1.),
        Random_Scaling(scaling_range=(0.95, 1.05), p=1.),
        Random_Rotate_Individual_Box(limit_angle=np.pi / 10, p=1.)
    ], p=1.)

    dataset = KittiDataset(configs, mode='val', lidar_aug=lidar_aug, aug_transforms=None,
                           num_samples=configs.num_samples)

    print('\n\nPress n to see the next sample >>> Press Esc to quit...')
    for idx in range(len(dataset)):
        if idx > 3:
            break
        lidarData, gt_boxes3d, cls_ids = dataset.draw_img_with_label(idx)
        # lidarData, gt_boxes3d, cls_ids, voxels_features, voxels_coors, targets = dataset.load_img_with_targets(idx)

        # view in point cloud
        fig = draw_lidar(lidarData, is_grid=False, is_top_region=True)
        draw_gt_boxes3d(gt_boxes3d=gt_boxes3d, fig=fig, cls_ids=cls_ids)
        # mlab.savefig(filename='test.png')
        mlab.show()
        if cv2.waitKey(0) & 0xff == 27:
            break
