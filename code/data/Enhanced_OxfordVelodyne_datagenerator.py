import os
import sys
import torch
import numpy as np
import pickle
import os.path as osp
import h5py
import json
import MinkowskiEngine as ME
from data.robotcar_sdk.python.interpolate_poses import interpolate_ins_poses, interpolate_vo_poses
from data.robotcar_sdk.python.transform import build_se3_transform
from data.robotcar_sdk.python.velodyne import load_velodyne_binary_seg
from torch.utils import data
from utils.pose_util import calibrate_process_poses, filter_overflow_ts
from copy import deepcopy

BASE_DIR = osp.dirname(osp.abspath(__file__))


class RobotCar(data.Dataset):
    def __init__(self, data_path, train=True, valid=False, augmentation=[],  voxel_size=0.3, real=False,
                 vo_lib='stereo', num_grid_x=0, num_grid_y=0, block_num=1):
        # directories
        lidar = 'velodyne_left'
        data_dir = osp.join(data_path, 'Oxford')

        # decide which sequences to use
        if train:
            split_filename = osp.join(data_dir, 'train_split.txt')
        elif valid:
            split_filename = osp.join(data_dir, 'valid_split.txt')
        else:
            split_filename = osp.join(data_dir, 'test_split.txt')
        with open(split_filename, 'r') as f:
            seqs = [l.rstrip() for l in f if not l.startswith('#')]

        ps = {}
        ts = {}
        vo_stats = {}
        pcs_all = []
        self.pcs = []

        for seq in seqs:
            seq_dir = osp.join(data_dir, seq + '-radar-oxford-10k')
            # read the image timestamps
            h5_path = osp.join(seq_dir, lidar + '_calibrate' + str(real) + '.h5')
            if not os.path.isfile(h5_path):
                print('interpolate ' + seq)
                ts_filename = osp.join(seq_dir, lidar + '.timestamps')
                with open(ts_filename, 'r') as f:
                    ts_raw = [int(l.rstrip().split(' ')[0]) for l in f]
                ins_filename = osp.join(seq_dir, 'gps', 'ins.csv')
                ts[seq] = filter_overflow_ts(ins_filename, ts_raw)
                rot = np.fromfile(osp.join(seq_dir, 'rot_tr.bin'), dtype=np.float32).reshape(-1, 9)
                t   = np.fromfile(osp.join(seq_dir, 'tr_add_mean.bin'), dtype=np.float32).reshape(-1, 3)
                ps[seq] = np.concatenate((rot, t), axis=1) # (n, 12)
                # write to h5 file
                print('write interpolate pose to ' + h5_path)
                h5_file = h5py.File(h5_path, 'w')
                h5_file.create_dataset('valid_timestamps', data=np.asarray(ts[seq], dtype=np.int64))
                h5_file.create_dataset('poses', data=ps[seq])
            else:
                # load h5 file, save pose interpolating time
                print("load " + seq + ' pose from ' + h5_path)
                h5_file = h5py.File(h5_path, 'r')
                ts[seq] = h5_file['valid_timestamps'][...]
                ps[seq] = h5_file['poses'][...]
            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

            pcs_all.extend([osp.join(seq_dir, 'SPVNAS_velodyne_left_plane_segmented', '{:d}.bin'.format(t)) for t in ts[seq]])

        # read / save pose normalization information
        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))
        pose_stats_filename = osp.join(data_dir, 'pose_stats_PGO.txt')
        if train:
            mean_t = np.mean(poses[:, 9:], axis=0)  # (3,)
            std_t = np.std(poses[:, 9:], axis=0)  # (3,)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        # convert the pose to translation + log quaternion, align, normalize
        self.poses = np.empty((0, 6))
        self.rots = np.empty((0, 3, 3))
        self.poses_max = np.empty((0, 2))
        self.poses_min = np.empty((0, 2))
        poses_all = np.empty((0, 6))
        rots_all = np.empty((0, 3, 3))

        pose_max_min_filename = osp.join(data_dir, 'pose_max_min.txt')

        for seq in seqs:
            pss, rotation, pss_max, pss_min = calibrate_process_poses(poses_in=ps[seq], mean_t=mean_t,
                                                            align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                                                            align_s=vo_stats[seq]['s'])
            poses_all = np.vstack((poses_all, pss))
            self.poses_max = np.vstack((self.poses_max, pss_max))
            self.poses_min = np.vstack((self.poses_min, pss_min))
            rots_all = np.vstack((rots_all, rotation))

        if train:
            # self.poses_max = np.max(self.poses_max, axis=0)  # (2,)
            # self.poses_min = np.min(self.poses_min, axis=0)  # (2,)
            # self.poses_max = np.max(self.poses_max, axis=0) * std_t[:2] + mean_t[:2]
            # self.poses_min = np.min(self.poses_min, axis=0) * std_t[:2] + mean_t[:2]
            self.poses_max = np.max(self.poses_max, axis=0) + mean_t[:2]
            self.poses_min = np.min(self.poses_min, axis=0) + mean_t[:2]
            block_size = list((np.array(list(self.poses_max)) - np.array(list(self.poses_min))) / block_num)
            # center_point = list((np.array(list(self.poses_min)) + np.array(list(self.poses_max))) / 2)
            np.savetxt(pose_max_min_filename, np.vstack((self.poses_max, self.poses_min)), fmt='%8.7f')
        else:
            self.poses_max, self.poses_min = np.loadtxt(pose_max_min_filename)
            block_size = list((np.array(list(self.poses_max)) - np.array(list(self.poses_min))) / block_num)
            # center_point = list((np.array(list(self.poses_min)) + np.array(list(self.poses_max))) / 2)
        # poses_all_real = poses_all[:, :2] * std_t[:2] + mean_t[:2] - self.poses_min
        poses_all_real = poses_all[:, :2] + mean_t[:2] - self.poses_min
        # divide the area into subregions
        if block_num!=1:
            for i in range(len(poses_all)):
                if int((poses_all_real[i, 0]) / block_size[0]) == num_grid_x and int(
                        poses_all_real[i, 1] / block_size[1]) == num_grid_y:
                    self.poses = np.vstack((self.poses, poses_all[i]))
                    self.pcs.append(pcs_all[i])
                    self.rots = np.vstack((self.rots, rots_all[i].reshape(1, 3, 3)))
        else:
            self.poses = poses_all
            self.pcs   = pcs_all
            self.rots  = rots_all


        self.augmentation = augmentation
        self.voxel_size = voxel_size

        if train:
            print("train data num:" + str(len(self.poses)))
        else:
            print("valid data num:" + str(len(self.poses)))

    def __getitem__(self, index):
        scan_path = self.pcs[index]
        ptcld = load_velodyne_binary_seg(scan_path)  # (N, 4)
        # print(ptcld.shape)
        # scan = np.concatenate((ptcld[:,0].reshape(-1, 1), ptcld[:,1].reshape(-1,1), -1*ptcld[:,2].reshape(-1,1)), axis=1)
        scan = ptcld[:, :3]  # (N, 3)
        scan = np.ascontiguousarray(scan)
        label = ptcld[:, 3].reshape(-1, 1)
        label = np.ascontiguousarray(label)
        for a in self.augmentation:
            scan = a.apply(scan)

        pose = self.poses[index]  # (6,)
        rot = self.rots[index]
        # ground truth
        scan_gt = (rot @ scan.transpose(1, 0)).transpose(1, 0) + pose[:3].reshape(1, 3)
        scan_gt_s8 = np.concatenate((scan, scan_gt, label), axis=1)

        coords, feats= ME.utils.sparse_quantize(
            coordinates=scan,
            features=scan,
            quantization_size=self.voxel_size)

        coords_s8, feats_s8 = ME.utils.sparse_quantize(
            coordinates=scan,
            features=scan_gt_s8,
            quantization_size=self.voxel_size*8)

        return (coords, feats, coords_s8, feats_s8, rot, pose)

    def __len__(self):
        return len(self.poses)