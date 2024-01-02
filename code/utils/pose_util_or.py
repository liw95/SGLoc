import numpy as np
import pandas as pd
import transforms3d.quaternions as txq
import transforms3d.euler as euler
import struct
import open3d
import torch
import scipy.interpolate
from scipy.spatial.transform import Rotation as R
from os import path as osp


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part last, as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )

    return o.reshape(quaternions.shape[:-1] + (3, 3))

def qlog(q):
    """
    Applies logarithm map to q
    :param q: (4,)
    :return: (3,)
    """
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])

    return q


def qexp(q):
    """
    Applies the exponential map to q
    :param q: (3,)
    :return: (4,)
    """
    n = np.linalg.norm(q)
    q = np.hstack((np.cos(n), np.sinc(n/np.pi)*q))

    return q

def qexp_t(q):
    """
    Applies exponential map to log quaternion
    :param q: N x 3
    :return: N x 4
    """
    n = torch.norm(q, p=2, dim=1, keepdim=True)
    n = torch.clamp(n, min=1e-8)
    q = q * torch.sin(n)
    q = q / n
    q = torch.cat((torch.cos(n), q), dim=1)

    return q



def process_poses(poses_in, mean_t, std_t, align_R, align_t, align_s):
    poses_out = np.zeros((len(poses_in), 6))
    rot_out   = np.zeros((len(poses_in), 3, 3))
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]

    # align
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]
        rot_out[i, :, :] = R
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # constrain to hemisphere
        q = qlog(q)
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    # normalize translation
    poses_out[:, :3] -= mean_t
    # poses_out[:, :3] /= std_t

    # max and min
    pose_max = np.max(poses_out[:, :2], axis=0)  # (2,)
    pose_min = np.min(poses_out[:, :2], axis=0)  # (2,)

    return poses_out, rot_out, pose_max, pose_min

def process_poses_millimetters(poses_in, mean_t, std_t, align_R, align_t, align_s):
    poses_out = np.zeros((len(poses_in), 6))
    rot_out   = np.zeros((len(poses_in), 3, 3))
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]

    # align
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]
        rot_out[i, :, :] = R
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # constrain to hemisphere
        q = qlog(q)
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    # normalize translation
    poses_out[:, :3] = poses_out[:, :3]*1000
    # poses_out[:, :3] /= std_t

    # max and min
    pose_max = np.max(poses_out[:, :2], axis=0)  # (2,)
    pose_min = np.min(poses_out[:, :2], axis=0)  # (2,)

    return poses_out, rot_out, pose_max, pose_min

def val_translation(pred_p, gt_p):
    """
    test model, compute error (numpy)
    input:
        pred_p: [3,]
        gt_p: [3,]
    returns:
        translation error (m):
    """
    if isinstance(pred_p, np.ndarray):
        predicted   = pred_p
        groundtruth = gt_p
    else:
        predicted   = pred_p.cpu().numpy()
        groundtruth = gt_p.cpu().numpy()
    error = np.linalg.norm(groundtruth - predicted)

    return error


def val_rotation(pred_q, gt_q):
    """
    test model, compute error (numpy)
    input:
        pred_q: [4,]
        gt_q: [4,]
    returns:
        rotation error (degrees):
    """
    if isinstance(pred_q, np.ndarray):
        predicted   = pred_q
        groundtruth = gt_q
    else:
        predicted   = pred_q.cpu().numpy()
        groundtruth = gt_q.cpu().numpy()

    # d = abs(np.sum(np.multiply(groundtruth, predicted)))
    # if d != d:
    #     print("d is nan")
    #     raise ValueError
    # if d > 1:
    #     d = 1
    # error = 2 * np.arccos(d) * 180 / np.pi0
    # d     = abs(np.dot(groundtruth, predicted))
    # d     = min(1.0, max(-1.0, d))

    d     = np.abs(np.dot(groundtruth, predicted))
    d     = np.minimum(1.0, np.maximum(-1.0, d))
    error = 2 * np.arccos(d) * 180 / np.pi

    return error


def poses2mats(poses_in):
    poses_out = np.zeros((len(poses_in), 3, 3))  # (B, 3, 3)
    poses_qua = np.asarray([qexp(q) for q in poses_in.cpu().detach().numpy()])

    # align
    for i in range(len(poses_out)):
        R = txq.quat2mat(poses_qua[i])
        poses_out[i, ...] = R

    return poses_out

def estimate_poses(source_pc, target_pc, threshold=0.6):
    # print(source_pc.shape)
    num_points = source_pc.shape[0]
    pred_t = np.zeros((1, 3))
    pred_q = np.zeros((1, 4))
    index1 = np.arange(0, num_points)
    index2 = np.arange(0, num_points)
    # np.random.shuffle(index1)
    index1 = np.expand_dims(index1, axis=1)
    index2 = np.expand_dims(index2, axis=1)
    corr = np.concatenate((index1, index2), axis=1)

    source_xyz = source_pc
    target_xyz = target_pc
    source = open3d.geometry.PointCloud()
    target = open3d.geometry.PointCloud()
    source.points = open3d.utility.Vector3dVector(source_xyz)
    target.points = open3d.utility.Vector3dVector(target_xyz)
    corres = open3d.utility.Vector2iVector(corr)

    M = open3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source,
        target,
        corres,
        threshold,
        open3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            open3d.pipelines.registration.
            CorrespondenceCheckerBasedOnEdgeLength(0.9),
            open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold)
        ],
        open3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    pred_t[0, :] = M.transformation[:3, 3:].squeeze()
    pred_q[0, :] = txq.mat2quat(M.transformation[:3, :3])

    return pred_t, pred_q

def R_t_estimate_poses(source_pc, target_pc, threshold=0.6):
    # print(source_pc.shape)
    num_points = source_pc.shape[0]
    pred_t = np.zeros((1, 3))
    pred_q = np.zeros((1, 4))
    index1 = np.arange(0, num_points)
    index2 = np.arange(0, num_points)
    # np.random.shuffle(index1)
    index1 = np.expand_dims(index1, axis=1)
    index2 = np.expand_dims(index2, axis=1)
    corr = np.concatenate((index1, index2), axis=1)

    source_xyz = source_pc
    target_xyz = target_pc
    source = open3d.geometry.PointCloud()
    target = open3d.geometry.PointCloud()
    source.points = open3d.utility.Vector3dVector(source_xyz)
    target.points = open3d.utility.Vector3dVector(target_xyz)
    corres = open3d.utility.Vector2iVector(corr)

    M = open3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source,
        target,
        corres,
        threshold,
        open3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            open3d.pipelines.registration.
            CorrespondenceCheckerBasedOnEdgeLength(0.9),
            open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold)
        ],
        open3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    pred_t[0, :] = M.transformation[:3, 3:].squeeze()
    pred_q[0, :] = txq.mat2quat(M.transformation[:3, :3])

    return pred_t, pred_q, M.transformation[:3, 3:], M.transformation[:3, :3]

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    Input:
        src: source points, [N, C]
        dst: target points, [M, C]
    Output:
        dist: per-point square distance, [N, M]
    """
    N, _ = src.shape
    M, _ = dst.shape
    dist = -2 * (src @ dst.transpose())
    dist += np.sum(src ** 2, -1).reshape(N, 1)
    dist += np.sum(dst ** 2, -1).reshape(1, M)
    return dist

def plane_estimate_poses(source_pc, target_pc, label, R, t, threshold=0.6):
    plane_index = np.squeeze(label ==1)
    source_plane = source_pc[plane_index, :]
    # 计算转换后的平面
    trans_plane = (R @ source_plane.transpose()).transpose() + t.reshape(1, 3)
    distance = square_distance(trans_plane, target_pc)
    mask = (np.min(distance, 0) <= 5)
    # print(np.sum(mask))
    mask_source_pc = source_pc[mask, :]
    mask_target_pc = target_pc[mask, :]
    pred_t, pred_q = estimate_poses(mask_source_pc, mask_target_pc, threshold)

    return pred_t, pred_q

def icp_estimate_pose(source_pc, target_pc, trans_mat_ransac, threshold=0.2):
    # source_pc: 模型输出预测点云
    # targert_pc: 模型输出预测点云*RANSAC求得的变换矩阵
    pred_t = np.zeros((1, 3))
    pred_q = np.zeros((1, 4))
    source_xyz = source_pc
    target_xyz = target_pc
    source = open3d.geometry.PointCloud()
    target = open3d.geometry.PointCloud()
    source.points = open3d.utility.Vector3dVector(source_xyz)
    target.points = open3d.utility.Vector3dVector(target_xyz)
    M = open3d.pipelines.registration.registration_icp(source,  target, threshold, np.eye(4),
                                               open3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                               open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    Final_M = M.transformation @ trans_mat_ransac
    pred_t[0, :] = Final_M[:3, 3:].squeeze()
    pred_q[0, :] = txq.mat2quat(Final_M[:3, :3])

    return pred_t, pred_q

def ds_pc(cloud, target_num):
    if cloud.shape[0] < target_num:
        # Add in artificial points if necessary
        print('Only %i out of %i required points in raw point cloud. Duplicating...' % (cloud.shape[0], target_num))
        num_to_pad = target_num - cloud.shape[0]
        index      = np.random.choice(cloud.shape[0], size=num_to_pad, replace=True)
        pad_points = cloud[index, :]
        cloud      = np.concatenate((cloud, pad_points), axis=0)

        return cloud
    else:
        index = np.random.choice(cloud.shape[0], size=target_num, replace=True)
        cloud = cloud[index, :]

        return cloud


def filter_overflow_ts(filename, ts_raw):
    file_data = pd.read_csv(filename)
    base_name = osp.basename(filename)

    if base_name.find('vo') > -1:
        ts_key = 'source_timestamp'
    else:
        ts_key = 'timestamp'

    pose_timestamps     = file_data[ts_key].values
    min_pose_timestamps = min(pose_timestamps)
    max_pose_timestamps = max(pose_timestamps)
    ts_filted   = [t for t in ts_raw if min_pose_timestamps < t < max_pose_timestamps]
    abandon_num = len(ts_raw) - len(ts_filted)
    print('abandom %d pointclouds that exceed the range of %s' % (abandon_num, filename))
    
    return ts_filted


def filter_overflow_nclt(gt_filename, ts_raw): # 滤波函数
    # gt_filename: GT对应的文件名
    # ts_raw: 原始数据集提供的点云时间戳
    ground_truth = np.loadtxt(gt_filename, delimiter=",")[1:,0]
    min_pose_timestamps = min(ground_truth)
    max_pose_timestamps = max(ground_truth)
    ts_filted = [t for t in ts_raw if min_pose_timestamps < t < max_pose_timestamps]
    abandon_num = len(ts_raw) - len(ts_filted)
    print('abandom %d pointclouds that exceed the range of %s' % (abandon_num, gt_filename))

    return ts_filted


def interpolate_pose_nclt(gt_filename, ts_raw): # 插值函数
    # gt_filename: GT对应文件名
    # ts_raw: 滤波后的点云时间戳
    ground_truth = np.loadtxt(gt_filename, delimiter=",")
    interp = scipy.interpolate.interp1d(ground_truth[1:, 0], ground_truth[1:, 1:], kind='nearest', axis=0)
    pose_gt = interp(ts_raw)
    print(pose_gt.shape)

    return pose_gt


def process_poses_nclt(poses_in, mean_t, std_t, align_R, align_t, align_s): 
    """
    processes the 1x6 raw pose from dataset by aligning and then normalizing
    :param poses_in: N x 6
    :param mean_t: 3
    :param std_t: 3
    :return: processed poses (translation + quaternion) N x 7
    """
    poses_out = poses_in
    # 欧拉角转换为四元数，归一化
    for i in range(len(poses_out)):
        # 数据集中GT为rpy顺序
        q = euler.euler2quat(poses_out[i, 3], poses_out[i, 4], poses_out[i, 5], 'rzyx')
        q *= np.sign(q)  # constrain to hemisphere
        q = qlog(q)
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    # normalize translation
    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t

    # max and min
    pose_max = np.max(poses_out[:, :2], axis=0)  # (2,)
    pose_min = np.min(poses_out[:, :2], axis=0)  # (2,)

    return poses_out, pose_max, pose_min


def convert_nclt(x_s, y_s, z_s): # 输入点云转换函数
    # 文档种提供的转换函数
    # 原文档返回为x, y, z，但在绘制可视化图时z取负，此处先取负
    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, -z


def load_velodyne_binary_nclt(filename): # 读入二进制点云
    f_bin = open(filename, "rb")
    hits = []
    while True:
        x_str = f_bin.read(2)
        if x_str == b'': #eof
            break
        x = struct.unpack('<H', x_str)[0]
        y = struct.unpack('<H', f_bin.read(2))[0]
        z = struct.unpack('<H', f_bin.read(2))[0]
        i = struct.unpack('B', f_bin.read(1))[0]
        l = struct.unpack('B', f_bin.read(1))[0]

        x, y, z = convert_nclt(x, y, z)

        hits += [[x, y ,z]]

    f_bin.close()

    hits = np.array(hits)

    return hits

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    # batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    # new_points = points[batch_indices, idx, :]
    new_points = torch.squeeze(points[idx, :])
    return new_points

