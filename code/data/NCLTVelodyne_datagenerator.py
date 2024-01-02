import os
import numpy as np
import os.path as osp
import h5py
import MinkowskiEngine as ME
from torch.utils import data
from utils.pose_util import filter_overflow_nclt, interpolate_pose_nclt, so3_to_euler_nclt, process_poses

BASE_DIR = osp.dirname(osp.abspath(__file__))


class NCLT(data.Dataset):
    def __init__(self, data_path, train=True, valid=False, augmentation=[],  voxel_size=0.3, real=False):
        # directories
        lidar = 'velodyne_left'
        data_dir = osp.join(data_path, 'NCLT')

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
        self.pcs = []

        for seq in seqs:
            seq_dir = osp.join(data_dir, seq )
            # read the image timestamps
            h5_path = osp.join(seq_dir, lidar + '_' + str(real) + '.h5')
            if not os.path.isfile(h5_path):
                print('interpolate ' + seq)
                ts_raw = []
                # with seg label
                vel = os.listdir(seq_dir + '/SPVNAS_velodyne_left_plane_segmented')
                """
                # without seg label
                # vel = os.listdir(seq_dir + '/velodyne_left')
                """
                for i in range(len(vel)):
                    ts_raw.append(int(vel[i][:-4]))
                ts_raw = sorted(ts_raw)
                # GT poses
                gt_filename = osp.join(seq_dir, 'groundtruth_'+ seq + '.csv')
                ts[seq] = filter_overflow_nclt(gt_filename, ts_raw)
                p = interpolate_pose_nclt(gt_filename, ts[seq])  # (n, 6)
                p = so3_to_euler_nclt(p)   # (n, 4, 4)
                ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))  # (n, 12)

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

            """
            # with seg label
            self.pcs.extend([osp.join(seq_dir, 'SPVNAS_velodyne_left_plane_segmented', '{:d}.bin'.format(t)) for t in ts[seq]])
            """
            # without seg label
            self.pcs.extend([osp.join(seq_dir, 'velodyne_left', '{:d}.bin'.format(t)) for t in ts[seq]])


        # read / save pose normalization information
        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))
        pose_stats_filename = osp.join(data_dir, 'NCLT_pose_stats.txt')
        if train:
            mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)  # (3,)
            std_t = np.std(poses[:, [3, 7, 11]], axis=0)  # (3,)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        # convert the pose to translation + log quaternion, align, normalize
        self.poses = np.empty((0, 6))
        self.rots = np.empty((0, 3, 3))
        for seq in seqs:
            pss, rotation, pss_max, pss_min = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                                                            align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                                                            align_s=vo_stats[seq]['s'])
            self.poses = np.vstack((self.poses, pss))
            self.rots = np.vstack((self.rots, rotation))

        self.augmentation = augmentation
        self.voxel_size = voxel_size

        if train:
            print("train data num:" + str(len(self.poses)))
        else:
            print("valid data num:" + str(len(self.poses)))

    def __getitem__(self, index):
        scan_path = self.pcs[index]
        """
        # with seg label
        ptcld = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 5)[:, :4]  # (N, 4)
        """
        # without seg label
        ptcld = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)  # (N, 4)
        ptcld[:, 2] = -1 * ptcld[:, 2]
        ptcld[:, 3] = 1

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
