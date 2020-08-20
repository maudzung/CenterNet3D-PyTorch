"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: Testing script
"""

import argparse
import sys
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np
import mayavi.mlab as mlab

sys.path.append('../')

import config.kitti_config as cnf
from data_process.kitti_dataloader import create_test_dataloader
from data_process.kitti_data_utils import box3d_center_to_conners
from models.model_utils import create_model
from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import centernet3d_decode, post_processing, get_final_pred
from utils.torch_utils import _sigmoid
from utils.visualization_utils import draw_lidar, draw_gt_boxes3d


def parse_test_configs():
    parser = argparse.ArgumentParser(description='Demonstration config for CenterNet3D Implementation')
    parser.add_argument('--saved_fn', type=str, default='centernet3d', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='centernet3d', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str, default=None, metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--K', type=int, default=100,
                        help='the number of top K')

    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=None, type=int,
                        help='GPU index to use.')

    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--peak_thresh', type=float, default=0.2)

    parser.add_argument('--show_image', action='store_true',
                        help='If true, show the image during demostration')
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_centernet3d', metavar='PATH',
                        help='the video filename if the output format is video')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False  # For testing on 1 GPU only

    configs.sparse_shape = (40, 1400, 1600)
    configs.input_size = (1400, 1600)
    configs.hm_size = (350, 400)
    configs.down_ratio = 4
    configs.max_objects = 50

    configs.head_conv = 64
    configs.num_classes = 1
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos
    configs.num_conners = 4

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim,
        'hm_conners': configs.num_classes  # equal classes --> 3
    }
    configs.num_input_features = 4

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.root_dir = '../'
    configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'kitti')

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.root_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    return configs


if __name__ == '__main__':
    configs = parse_test_configs()

    model = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)

    out_cap = None

    model.eval()

    test_dataloader = create_test_dataloader(configs)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            sample_ids, batch_size, lidarData, voxels_features, voxels_coors = batch_data
            voxels_features = voxels_features.to(configs.device, non_blocking=True)
            t1 = time_synchronized()
            outputs = model(voxels_features, voxels_coors, batch_size)
            t2 = time_synchronized()
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['hm_conners'] = _sigmoid(outputs['hm_conners'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            outputs['direction'] = _sigmoid(outputs['direction'])
            # detections size (batch_size, K, 10)
            detections = centernet3d_decode(outputs['hm_cen'], outputs['hm_conners'], outputs['cen_offset'],
                                            outputs['direction'], outputs['z_coor'], outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy()
            detections = post_processing(detections, configs.num_classes, configs.down_ratio)
            detections = get_final_pred(detections[0], configs.num_classes, configs.peak_thresh)
            car_detections = detections[0]
            # calib = kitti_data_utils.Calibration(img_paths[0].replace(".png", ".txt").replace("image_2", "calib"))
            if len(car_detections) > 0:
                # Draw prediction in the image
                pred_boxes3d = []
                for car_det in car_detections[:, 1:]:
                    pred_boxes3d.append(box3d_center_to_conners(car_det))

                fig = draw_lidar(lidarData[0], is_grid=False, is_top_region=True)
                cls_ids = np.zeros((len(car_detections)), dtype=np.int)
                draw_gt_boxes3d(gt_boxes3d=pred_boxes3d, fig=fig, cls_ids=cls_ids)
                # mlab.savefig(filename='test.png')
                mlab.show()

            print('\tDone testing the {}th sample, time: {:.1f}ms, speed {:.2f}FPS'.format(batch_idx, (t2 - t1) * 1000,
                                                                                           1 / (t2 - t1)))