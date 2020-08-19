"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: the script for point clouds transformation
"""

import sys

import numpy as np
import cv2

sys.path.append('../')

import config.kitti_config as cnf


class Compose(object):
    def __init__(self, transforms, p=1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, lidar, gt_boxes3d):
        if np.random.random() <= self.p:
            for t in self.transforms:
                lidar, gt_boxes3d = t(lidar, gt_boxes3d)
        return lidar, gt_boxes3d


class OneOf(object):
    def __init__(self, transforms, p=1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, lidar, gt_boxes3d):
        if np.random.random() <= self.p:
            choice = np.random.randint(low=0, high=len(self.transforms))
            lidar, gt_boxes3d = self.transforms[choice](lidar, gt_boxes3d)

        return lidar, gt_boxes3d


class Random_Rotate_Individual_Box(object):
    def __init__(self, limit_angle=np.pi / 10, p=0.5):
        # -np.pi / 10, np.pi / 10
        self.limit_angle = limit_angle
        self.p = p

    def __call__(self, lidar, gt_boxes3d):
        """
        :param gt_boxes3d: # (N, 8, 3)
        :return:
        """
        if np.random.random() <= self.p:
            for idx in range(len(gt_boxes3d)):
                is_collision = True
                _count = 0
                while is_collision and _count < 100:
                    t_rz = np.random.uniform(-self.limit_angle, self.limit_angle)
                    t_x = np.random.normal()
                    t_y = np.random.normal()
                    t_z = np.random.normal()

                    # check collision
                    tmp = box_transform(gt_boxes3d[[idx]], t_x, t_y, t_z, t_rz)
                    is_collision = False
                    for idy in range(idx):
                        iou = cal_iou2d(tmp[0, :4, :2], gt_boxes3d[idy, :4, :2])
                        if iou > 0:
                            is_collision = True
                            _count += 1
                            break
                if not is_collision:
                    box_corner = gt_boxes3d[idx]
                    minx = np.min(box_corner[:, 0])
                    miny = np.min(box_corner[:, 1])
                    minz = np.min(box_corner[:, 2])
                    maxx = np.max(box_corner[:, 0])
                    maxy = np.max(box_corner[:, 1])
                    maxz = np.max(box_corner[:, 2])
                    bound_x = np.logical_and(lidar[:, 0] >= minx, lidar[:, 0] <= maxx)
                    bound_y = np.logical_and(lidar[:, 1] >= miny, lidar[:, 1] <= maxy)
                    bound_z = np.logical_and(lidar[:, 2] >= minz, lidar[:, 2] <= maxz)
                    bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)
                    lidar[bound_box, 0:3] = point_transform(lidar[bound_box, 0:3], t_x, t_y, t_z, rz=t_rz)
                    gt_boxes3d[idx] = box_transform(gt_boxes3d[[idx]], t_x, t_y, t_z, t_rz)

        return lidar, gt_boxes3d


class Random_Rotation(object):
    def __init__(self, limit_angle=np.pi / 4, p=0.5):
        self.limit_angle = limit_angle
        self.p = p

    def __call__(self, lidar, gt_boxes3d):
        """
        :param gt_boxes3d: # (N, 8, 3)
        :return:
        """
        if np.random.random() <= self.p:
            angle = np.random.uniform(-self.limit_angle, self.limit_angle)
            lidar[:, 0:3] = point_transform(lidar[:, 0:3], 0, 0, 0, rz=angle)
            gt_boxes3d = box_transform(gt_boxes3d, 0, 0, 0, r=angle)

        return lidar, gt_boxes3d


class Random_Scaling(object):
    def __init__(self, scaling_range=(0.95, 1.05), p=0.5):
        self.scaling_range = scaling_range
        self.p = p

    def __call__(self, lidar, gt_boxes3d):
        """
        :param gt_boxes3d: # (N, 8, 3)
        :return:
        """
        if np.random.random() <= self.p:
            factor = np.random.uniform(self.scaling_range[0], self.scaling_range[0])
            lidar[:, 0:3] = lidar[:, 0:3] * factor
            gt_boxes3d = gt_boxes3d * factor

        return lidar, gt_boxes3d


def box3d_cam_to_velo(box3d, Tr):
    def project_cam2velo(cam, Tr):
        T = np.zeros([4, 4], dtype=np.float32)
        T[:3, :] = Tr
        T[3, 3] = 1
        T_inv = np.linalg.inv(T)
        lidar_loc_ = np.dot(T_inv, cam)
        lidar_loc = lidar_loc_[:3]
        return lidar_loc.reshape(1, 3)

    def ry_to_rz(ry):
        angle = -ry - np.pi / 2

        if angle >= np.pi:
            angle -= np.pi
        if angle < -np.pi:
            angle = 2 * np.pi + angle

        return angle

    h, w, l, tx, ty, tz, ry = [float(i) for i in box3d]
    cam = np.ones([4, 1])
    cam[0] = tx
    cam[1] = ty
    cam[2] = tz
    t_lidar = project_cam2velo(cam, Tr)

    Box = np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                    [0, 0, 0, 0, h, h, h, h]])

    rz = ry_to_rz(ry)

    rotMat = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz), np.cos(rz), 0.0],
        [0.0, 0.0, 1.0]])

    velo_box = np.dot(rotMat, Box)

    cornerPosInVelo = velo_box + np.tile(t_lidar, (8, 1)).T

    box3d_corner = cornerPosInVelo.transpose()

    return box3d_corner.astype(np.float32)


def box3d_velo_to_velo(box3d):
    h, w, l, tx, ty, tz, rz = [float(i) for i in box3d]
    t_lidar = np.array([tx, ty, tz])
    Box = np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                    [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]])
    rotMat = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz), np.cos(rz), 0.0],
        [0.0, 0.0, 1.0]])
    velo_box = np.dot(rotMat, Box)
    cornerPosInVelo = velo_box + np.tile(t_lidar, (8, 1)).T
    box3d_corner = cornerPosInVelo.transpose()

    return box3d_corner.astype(np.float32)


def point_transform(points, tx, ty, tz, rx=0, ry=0, rz=0):
    # Input:
    #   points: (N, 3)
    #   rx/y/z: in radians
    # Output:
    #   points: (N, 3)
    N = points.shape[0]
    points = np.hstack([points, np.ones((N, 1))])
    mat1 = np.eye(4)
    mat1[3, 0:3] = tx, ty, tz
    points = np.matmul(points, mat1)
    if rx != 0:
        mat = np.zeros((4, 4))
        mat[0, 0] = 1
        mat[3, 3] = 1
        mat[1, 1] = np.cos(rx)
        mat[1, 2] = -np.sin(rx)
        mat[2, 1] = np.sin(rx)
        mat[2, 2] = np.cos(rx)
        points = np.matmul(points, mat)
    if ry != 0:
        mat = np.zeros((4, 4))
        mat[1, 1] = 1
        mat[3, 3] = 1
        mat[0, 0] = np.cos(ry)
        mat[0, 2] = np.sin(ry)
        mat[2, 0] = -np.sin(ry)
        mat[2, 2] = np.cos(ry)
        points = np.matmul(points, mat)
    if rz != 0:
        mat = np.zeros((4, 4))
        mat[2, 2] = 1
        mat[3, 3] = 1
        mat[0, 0] = np.cos(rz)
        mat[0, 1] = -np.sin(rz)
        mat[1, 0] = np.sin(rz)
        mat[1, 1] = np.cos(rz)
        points = np.matmul(points, mat)
    return points[:, 0:3]


def box_transform(boxes_corner, tx, ty, tz, r=0):
    # boxes_corner (N, 8, 3)
    for idx in range(len(boxes_corner)):
        boxes_corner[idx] = point_transform(boxes_corner[idx], tx, ty, tz, rz=r)
    return boxes_corner


def cal_iou2d(box1_corner, box2_corner):
    box1_corner = np.reshape(box1_corner, [4, 2])
    box2_corner = np.reshape(box2_corner, [4, 2])
    box1_corner = ((cnf.W, cnf.H) - (box1_corner - (cnf.boundary['minX'], cnf.boundary['minY'])) / (
        cnf.vw, cnf.vh)).astype(np.int32)
    box2_corner = ((cnf.W, cnf.H) - (box2_corner - (cnf.boundary['minX'], cnf.boundary['minY'])) / (
        cnf.vw, cnf.vh)).astype(np.int32)

    buf1 = np.zeros((cnf.H, cnf.W, 3))
    buf2 = np.zeros((cnf.H, cnf.W, 3))
    buf1 = cv2.fillConvexPoly(buf1, box1_corner, color=(1, 1, 1))[..., 0]
    buf2 = cv2.fillConvexPoly(buf2, box2_corner, color=(1, 1, 1))[..., 0]

    indiv = np.sum(np.absolute(buf1 - buf2))
    share = np.sum((buf1 + buf2) == 2)
    if indiv == 0:
        return 0.0  # when target is out of bound

    return share / (indiv + share)
