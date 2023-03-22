"""
This script is borrowed from https://github.com/mkocabas/VIBE.
Adhere to their license to use this script.

We hacked it a little bit to make it happy in our framework.
"""

import sys
sys.path.append('.')
import os
import cv2
import h5py
import glob
import json
import joblib
import argparse
import numpy as np
from tqdm import tqdm
import os.path as osp
import scipy.io as sio

from lib.core.config import DB_DIR, MPII3D_DIR
from lib.utils.utils import tqdm_enumerate
from lib.data_utils.kp_utils import convert_kps
from lib.data_utils.img_utils import get_bbox_from_kp2d


def read_openpose(json_file, gt_part, dataset):
    # get only the arms/legs joints
    op_to_12 = [11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7]
    # read the openpose detection
    json_data = json.load(open(json_file, 'r'))
    people = json_data['people']
    if len(people) == 0:
        # no openpose detection
        keyp25 = np.zeros([25,3])
    else:
        # size of person in pixels
        scale = max(max(gt_part[:,0])-min(gt_part[:,0]),max(gt_part[:,1])-min(gt_part[:,1]))
        # go through all people and find a match
        dist_conf = np.inf*np.ones(len(people))
        for i, person in enumerate(people):
            # openpose keypoints
            op_keyp25 = np.reshape(person['pose_keypoints_2d'], [25,3])
            op_keyp12 = op_keyp25[op_to_12, :2]
            op_conf12 = op_keyp25[op_to_12, 2:3] > 0
            # all the relevant joints should be detected
            if min(op_conf12) > 0:
                # weighted distance of keypoints
                dist_conf[i] = np.mean(np.sqrt(np.sum(op_conf12*(op_keyp12 - gt_part[:12, :2])**2, axis=1)))
        # closest match
        p_sel = np.argmin(dist_conf)
        # the exact threshold is not super important but these are the values we used
        if dataset == 'mpii':
            thresh = 30
        elif dataset == 'coco':
            thresh = 10
        else:
            thresh = 0
        # dataset-specific thresholding based on pixel size of person
        if min(dist_conf)/scale > 0.1 and min(dist_conf) < thresh:
            keyp25 = np.zeros([25,3])
        else:
            keyp25 = np.reshape(people[p_sel]['pose_keypoints_2d'], [25,3])
    return keyp25


def read_calibration(calib_file, vid_list):
    Ks, Rs, Ts = [], [], []
    file = open(calib_file, 'r')
    content = file.readlines()
    for vid_i in vid_list:
        K = np.array([float(s) for s in content[vid_i * 7 + 5][11:-2].split()])
        K = np.reshape(K, (4, 4))
        RT = np.array([float(s) for s in content[vid_i * 7 + 6][11:-2].split()])
        RT = np.reshape(RT, (4, 4))
        R = RT[:3, :3]
        T = RT[:3, 3] / 1000
        Ks.append(K)
        Rs.append(R)
        Ts.append(T)
    return Ks, Rs, Ts


def read_data_train(dataset_path, user_list, seq_list, vid_list, debug=False):
    h, w = 2048, 2048
    dataset = {
        'vid_name': [],
        'frame_id': [],
        'joints3D': [],
        'joints2D': [],
        'bbox': [],
        'img_name': [],
    }

    for user_i in user_list:
        for seq_i in seq_list:
            seq_path = os.path.join(dataset_path,
                                    'S' + str(user_i),
                                    'Seq' + str(seq_i))
            # mat file with annotations
            annot_file = os.path.join(seq_path, 'annot.mat')
            annot2 = sio.loadmat(annot_file)['annot2']
            annot3 = sio.loadmat(annot_file)['annot3']
            # calibration file and camera parameters
            for j, vid_i in enumerate(vid_list):
                # image folder
                imgs_path = os.path.join(seq_path,
                                         'video_' + str(vid_i))
                # print(annot2, annot3, imgs_path)
                # per frame
                
                
                if not os.path.isdir(imgs_path):
                    continue
                pattern = os.path.join(imgs_path, '*.jpg')
                img_list = sorted(glob.glob(pattern))
                vid_used_frames = []
                vid_used_joints = []
                vid_used_bbox = []
                vid_segments = []
                vid_uniq_id = "subj" + str(user_i) + '_seq' + str(seq_i) + "_vid" + str(vid_i) + "_seg0"
                for i, img_i in tqdm_enumerate(img_list, desc="sub{}_seq{}_vid{}".format(user_i, seq_i, vid_i)):

                    # for each image we store the relevant annotations
                    img_name = img_i.split('/')[-1]
                    joints_2d_raw = np.reshape(annot2[vid_i][0][i], (1, 28, 2))
                    joints_2d_raw= np.append(joints_2d_raw, np.ones((1,28,1)), axis=2)
                    joints_2d = convert_kps(joints_2d_raw, "mpii3d",  "spin").reshape((-1,3))

                    joints_3d_raw = np.reshape(annot3[vid_i][0][i], (1, 28, 3)) / 1000
                    joints_3d = convert_kps(joints_3d_raw, "mpii3d", "spin").reshape((-1,3))

                    bbox = get_bbox_from_kp2d(joints_2d[~np.all(joints_2d == 0, axis=1)]).reshape(4)

                    joints_3d = joints_3d - joints_3d[39]  # 4 is the root

                    # check that all joints are visible
                    x_in = np.logical_and(joints_2d[:, 0] < w, joints_2d[:, 0] >= 0)
                    y_in = np.logical_and(joints_2d[:, 1] < h, joints_2d[:, 1] >= 0)
                    ok_pts = np.logical_and(x_in, y_in)
                    
                    if np.sum(ok_pts) < joints_2d.shape[0]:
                        vid_uniq_id = "_".join(vid_uniq_id.split("_")[:-1])+ "_seg" +\
                                          str(int(dataset['vid_name'][-1].split("_")[-1][3:])+1)
                        continue
                    
                    dataset['vid_name'].append(vid_uniq_id)
                    dataset['frame_id'].append(img_name.split(".")[0])
                    dataset['img_name'].append(img_i)
                    dataset['joints2D'].append(joints_2d)
                    dataset['joints3D'].append(joints_3d)
                    dataset['bbox'].append(bbox)
                    vid_segments.append(vid_uniq_id)
                    vid_used_frames.append(img_i)
                    vid_used_joints.append(joints_2d)
                    vid_used_bbox.append(bbox)

                vid_segments= np.array(vid_segments)
                ids = np.zeros((len(set(vid_segments))+1))
                ids[-1] = len(vid_used_frames) + 1
                if (np.where(vid_segments[:-1] != vid_segments[1:])[0]).size != 0:
                    ids[1:-1] = (np.where(vid_segments[:-1] != vid_segments[1:])[0]) + 1


    for k in dataset.keys():
        dataset[k] = np.array(dataset[k])

    valid = np.zeros([len(dataset['joints3D']), 49, 1])
    valid[:, 25:39, :] = 1
    valid[:, (39, 41, 43), :] = 1
    dataset['joints3D'] = np.concatenate([dataset['joints3D'], valid], axis=-1)

    return dataset


def read_test_data(dataset_path):

    dataset = {
        'vid_name': [],
        'frame_id': [],
        'joints3D': [],
        'joints2D': [],
        'bbox': [],
        'img_name': [],
        'features': [],
        "valid_i": []
    }

    user_list = range(1, 7)

    for user_i in user_list:
        print('Subject', user_i)
        seq_path = os.path.join(dataset_path,
                                'mpi_inf_3dhp_test_set',
                                'TS' + str(user_i))
        # mat file with annotations
        annot_file = os.path.join(seq_path, 'annot_data.mat')
        mat_as_h5 = h5py.File(annot_file, 'r')
        annot2 = np.array(mat_as_h5['annot2'])
        annot3 = np.array(mat_as_h5['univ_annot3'])
        valid = np.array(mat_as_h5['valid_frame'])

        vid_used_frames = []
        vid_used_joints = []
        vid_used_bbox = []
        vid_segments = []
        vid_uniq_id = "subj" + str(user_i) + "_seg0"


        for frame_i, valid_i in tqdm(enumerate(valid)):

            img_i = os.path.join('mpi_inf_3dhp_test_set',
                                    'TS' + str(user_i),
                                    'imageSequence',
                                    'img_' + str(frame_i + 1).zfill(6) + '.jpg')

            joints_2d_raw = np.expand_dims(annot2[frame_i, 0, :, :], axis = 0)
            joints_2d_raw = np.append(joints_2d_raw, np.ones((1, 17, 1)), axis=2)


            joints_2d = convert_kps(joints_2d_raw, src="mpii3d_test", dst="spin").reshape((-1, 3))

            joints_3d_raw = np.reshape(annot3[frame_i, 0, :, :], (1, 17, 3)) / 1000
            joints_3d = convert_kps(joints_3d_raw, "mpii3d_test", "spin").reshape((-1, 3))
            joints_3d = joints_3d - joints_3d[39] # substract pelvis zero is the root for test

            bbox = get_bbox_from_kp2d(joints_2d[~np.all(joints_2d == 0, axis=1)]).reshape(4)


            # check that all joints are visible
            img_file = os.path.join(dataset_path, img_i)
            I = cv2.imread(img_file)
            h, w, _ = I.shape
            x_in = np.logical_and(joints_2d[:, 0] < w, joints_2d[:, 0] >= 0)
            y_in = np.logical_and(joints_2d[:, 1] < h, joints_2d[:, 1] >= 0)
            ok_pts = np.logical_and(x_in, y_in)

            if np.sum(ok_pts) < joints_2d.shape[0]:
                vid_uniq_id = "_".join(vid_uniq_id.split("_")[:-1]) + "_seg" + \
                              str(int(dataset['vid_name'][-1].split("_")[-1][3:]) + 1)
                continue

            print(joints_3d.shape)
            dataset['vid_name'].append(vid_uniq_id)
            dataset['frame_id'].append(img_file.split("/")[-1].split(".")[0])
            dataset['img_name'].append(img_file)
            dataset['joints2D'].append(joints_2d)
            dataset['joints3D'].append(joints_3d)
            dataset['bbox'].append(bbox)
            dataset['valid_i'].append(valid_i)

            vid_segments.append(vid_uniq_id)
            vid_used_frames.append(img_file)
            vid_used_joints.append(joints_2d)
            vid_used_bbox.append(bbox)

        vid_segments = np.array(vid_segments)
        ids = np.zeros((len(set(vid_segments)) + 1))
        ids[-1] = len(vid_used_frames) + 1
        if (np.where(vid_segments[:-1] != vid_segments[1:])[0]).size != 0:
            ids[1:-1] = (np.where(vid_segments[:-1] != vid_segments[1:])[0]) + 1
        
    for k in dataset.keys():
        dataset[k] = np.array(dataset[k])

    valid = np.zeros([len(dataset['joints3D']), 49, 1])
    valid[:, 25:39, :] = 1
    valid[:, (39, 41, 43), :] = 1
    dataset['joints3D'] = np.concatenate([dataset['joints3D'], valid], axis=-1)

    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_dir', type=str, help='dataset directory', default=MPII3D_DIR)
    parser.add_argument('--out_dir', type=str, help='output directory', default=DB_DIR)
    parser.add_argument('--sub', nargs='+', type=int, default=[1,2,3,4,5,6,7,8])
    parser.add_argument('--seq', nargs='+', type=int, default=[1,2])
    parser.add_argument('--vid', nargs='+', type=int, default=[0,1,2,3,4,5,6,7,8])
    args = parser.parse_args()

    print(args.sub)
    print(args.seq)
    print(args.vid)

    dataset = read_data_train(args.inp_dir, args.sub, args.seq, args.vid)
    joblib.dump(dataset, osp.join(args.out_dir, 'mpii3d_train_db.pt'))

    #dataset = read_test_data(args.inp_dir)
    #joblib.dump(dataset, osp.join(args.out_dir, 'mpii3d_val_db.pt'))



