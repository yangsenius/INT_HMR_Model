# -*- coding: utf-8 -*-
"""
This script is borrowed from https://github.com/mkocabas/VIBE.
Adhere to their license to use this script.

We hacked it a little bit to make it happy in our framework.
"""

import sys
sys.path.append('.')

import os
import cv2
import torch
import joblib
import argparse
import numpy as np
import pickle as pkl
import os.path as osp
from tqdm import tqdm

from lib.data_utils.kp_utils import *
from lib.core.config import DB_DIR, DATA_DIR, THREEDPW_DIR
from lib.utils.smooth_bbox import get_smooth_bbox_params
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14
from lib.utils.geometry import batch_rodrigues, rotation_matrix_to_angle_axis
from lib.data_utils.kp_utils import convert_kps

NUM_JOINTS = 24
VIS_THRESH = 0.3
MIN_KP = 6

def read_data(folder, set, debug=False):

    dataset = {
        'vid_name': [],
        'frame_id': [],
        'joints3D': [],
        'joints2D': [],
        'shape': [],
        'pose': [],
        'bbox': [],
        'img_name': [],
        'valid': [],
    }

    sequences = [x.split('.')[0] for x in os.listdir(osp.join(folder, 'sequenceFiles', set))]

    J_regressor = None

    smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)
    if set == 'test' or set == 'validation':
        J_regressor = torch.from_numpy(np.load(osp.join(DATA_DIR, 'J_regressor_h36m.npy'))).float()

    for i, seq in tqdm(enumerate(sequences)):

        data_file = osp.join(folder, 'sequenceFiles', set, seq + '.pkl')

        data = pkl.load(open(data_file, 'rb'), encoding='latin1')

        img_dir = osp.join(folder, 'imageFiles', seq)

        num_people = len(data['poses'])
        num_frames = len(data['img_frame_ids'])
        assert (data['poses2d'][0].shape[0] == num_frames)

        for p_id in range(num_people):
            pose = torch.from_numpy(data['poses'][p_id]).float()
            shape = torch.from_numpy(data['betas'][p_id][:10]).float().repeat(pose.size(0), 1)
            trans = torch.from_numpy(data['trans'][p_id]).float()
            j2d = data['poses2d'][p_id].transpose(0,2,1)
            cam_pose = data['cam_poses']
            campose_valid = data['campose_valid'][p_id]

            # ======== Align the mesh params ======== #
            rot = pose[:, :3]
            rot_mat = batch_rodrigues(rot)

            Rc = torch.from_numpy(cam_pose[:, :3, :3]).float()
            Rs = torch.bmm(Rc, rot_mat.reshape(-1, 3, 3))
            rot = rotation_matrix_to_angle_axis(Rs)
            pose[:, :3] = rot
            # ======== Align the mesh params ======== #

            output = smpl(betas=shape, body_pose=pose[:,3:], global_orient=pose[:,:3], transl=trans)
            # verts = output.vertices
            j3d = output.joints

            if J_regressor is not None:
                vertices = output.vertices
                J_regressor_batch = J_regressor[None, :].expand(vertices.shape[0], -1, -1).to(vertices.device)
                j3d = torch.matmul(J_regressor_batch, vertices)
                j3d = j3d[:, H36M_TO_J14, :]

            img_paths = []
            for i_frame in range(num_frames):
                img_path = os.path.join(img_dir + '/image_{:05d}.jpg'.format(i_frame))
                img_paths.append(img_path)

            bbox_params, time_pt1, time_pt2 = get_smooth_bbox_params(j2d, vis_thresh=VIS_THRESH, sigma=8)

            # process bbox_params
            c_x = bbox_params[:,0]
            c_y = bbox_params[:,1]
            scale = bbox_params[:,2]
            w = h = 150. / scale
            w = h = h * 1.1
            bbox = np.vstack([c_x,c_y,w,h]).T

            # process keypoints
            j2d[:, :, 2] = j2d[:, :, 2] > 0.3  # set the visibility flags
            # Convert to common 2d keypoint format
            perm_idxs = get_perm_idxs('3dpw', 'common')
            perm_idxs += [0, 0]  # no neck, top head
            j2d = j2d[:, perm_idxs]
            j2d[:, 12:, 2] = 0.0

            # print('j2d', j2d[time_pt1:time_pt2].shape)
            # print('campose', campose_valid[time_pt1:time_pt2].shape)

            img_paths_array = np.array(img_paths)[time_pt1:time_pt2]
            dataset['vid_name'].append(np.array([f'{seq}_{p_id}']*num_frames)[time_pt1:time_pt2])
            dataset['frame_id'].append(np.arange(0, num_frames)[time_pt1:time_pt2])
            dataset['img_name'].append(img_paths_array)
            dataset['joints3D'].append(j3d.numpy()[time_pt1:time_pt2])
            dataset['joints2D'].append(j2d[time_pt1:time_pt2])
            dataset['shape'].append(shape.numpy()[time_pt1:time_pt2])
            dataset['pose'].append(pose.numpy()[time_pt1:time_pt2])
            dataset['bbox'].append(bbox)
            dataset['valid'].append(campose_valid[time_pt1:time_pt2])

    for k in dataset.keys():
        dataset[k] = np.concatenate(dataset[k])
        print(k, dataset[k].shape)

    # Filter out keypoints
    indices_to_use = np.where((dataset['joints2D'][:, :, 2] > VIS_THRESH).sum(-1) > MIN_KP)[0]
    for k in dataset.keys():
        dataset[k] = dataset[k][indices_to_use]

    dataset['joints2D'] = convert_kps(dataset['joints2D'], src='common', dst='spin')
    valid = np.zeros([len(dataset['joints3D']), 49, 1])
    valid[:, 25:39, :] = 1
    if set != 'train':
        dataset['joints3D'] = convert_kps(dataset['joints3D'], src='common', dst='spin')
    dataset['joints3D'] = np.concatenate([dataset['joints3D'], valid], axis=-1)
    
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_dir', type=str, help='dataset directory', default=THREEDPW_DIR)
    parser.add_argument('--out_dir', type=str, help='output directory', default=DB_DIR)
    args = parser.parse_args()

    debug = False

    dataset = read_data(args.inp_dir, 'validation', debug=debug)
    joblib.dump(dataset, osp.join(args.out_dir, '3dpw_val_db.pt'))

    dataset = read_data(args.inp_dir, 'train', debug=debug)
    joblib.dump(dataset, osp.join(args.out_dir, '3dpw_train_db.pt'))

    dataset = read_data(args.inp_dir, 'test', debug=debug)
    joblib.dump(dataset, osp.join(args.out_dir, '3dpw_test_db.pt'))
