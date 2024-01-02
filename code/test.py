# pylint: disable=no-member
import argparse
import os
import sys
import numpy as np
import random
import time
import torch
import MinkowskiEngine as ME
import matplotlib

# The quality-enhanced Oxford dataset
from data.QEOxfordVelodyne_datagenerator import QEOxford
# The Oxford dataset
from data.OxfordVelodyne_datagenerator import Oxford
# The NCLT dataset
from data.NCLTVelodyne_datagenerator import NCLT
from models.model import SGLoc
from data.base_loader import CollationFunctionFactory
from utils.pose_util import val_translation, val_rotation, qexp, estimate_pose
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.utils.data import DataLoader
from os import path as osp
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
cudnn.enabled = True
torch.set_num_threads(4)
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0,
                    help='gpu id for network, only effective when multi_gpus is false')
parser.add_argument('--val_batch_size', type=int, default=30,
                    help='Batch Size during validating [default: 80]')
parser.add_argument('--log_dir', default='log_Oxford/',
                    help='Log dir [default: log]')
parser.add_argument('--dataset_folder', default='/home/ldq/Codes/SGLoc',
                    help='Our Dataset Folder')
parser.add_argument('--seed', type=int, default=20, metavar='S',
                    help='random seed (default: 20)')
parser.add_argument('--dataset', default='NCLT',
                    help='Oxford or NCLT')
parser.add_argument('--num_workers', type=int, default=4,
                    help='num workers for dataloader, default:4')
parser.add_argument('--voxel_size', type=float, default=0.2,
                    help='Number of points to downsample model to')
parser.add_argument('--resume_model', type=str, default='Oxford_checkpoint.tar',
                    help='If present, restore checkpoint and resume training')


FLAGS = parser.parse_args()
args = vars(FLAGS)
for (k, v) in args.items():
    print('%s: %s' % (str(k), str(v)))
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)

LOG_FOUT = open(os.path.join(FLAGS.log_dir, 'log_test.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')
TOTAL_ITERATIONS = 0
NUM = 0

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_id)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise ValueError("GPU not found!")


valid_kwargs = dict(data_path=FLAGS.dataset_folder,
                    train=False,
                    valid=True,
                    voxel_size=FLAGS.voxel_size)


if FLAGS.dataset == 'Oxford':
    val_set = Oxford(**valid_kwargs)
    dataset = 'Oxford&QEOxford'
elif FLAGS.dataset == 'QEOxford':
    val_set = QEOxford(**valid_kwargs)
    dataset = 'Oxford&QEOxford'
elif FLAGS.dataset == 'NCLT':
    val_set = NCLT(**valid_kwargs)
    dataset = 'NCLT'
else:
    raise ValueError("dataset error!")

pose_stats_file = os.path.join(FLAGS.dataset_folder, dataset, FLAGS.dataset + '_pose_stats.txt')
pose_m, pose_s = np.loadtxt(pose_stats_file)

collation_fn = CollationFunctionFactory(collation_type='collate_pair')

val_loader = DataLoader(val_set,
                        batch_size=FLAGS.val_batch_size,
                        shuffle=False,
                        collate_fn=collation_fn,
                        num_workers=FLAGS.num_workers,
                        pin_memory=True)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def eval():
    global TOTAL_ITERATIONS
    global NUM
    setup_seed(FLAGS.seed)
    val_writer = SummaryWriter(os.path.join(FLAGS.log_dir, 'valid'))
    model = SGLoc()
    model = model.to(device)

    LOG_FOUT.write("\n")
    LOG_FOUT.flush()
    if len(FLAGS.resume_model) > 0:
        resume_filename = FLAGS.log_dir + FLAGS.resume_model
        print("Resuming From ", resume_filename)
        checkpoint = torch.load(resume_filename)
        saved_state_dict = checkpoint['state_dict']
        model.load_state_dict(saved_state_dict)
    sys.stdout.flush()
    # for xxx in range(10):
    for threshold in range(14, 15, 2):
        log_string('**** THRESHOLD %01f ****' % (threshold/10))
        valid_one_epoch(model, val_loader, val_writer, device, threshold/10)


def valid_one_epoch(model, val_loader, val_writer, device, threshold):
    gt_translation = np.zeros((len(val_set), 3))
    pred_translation = np.zeros((len(val_set), 3))
    gt_rotation = np.zeros((len(val_set), 4))
    pred_rotation = np.zeros((len(val_set), 4))

    error_t = np.zeros(len(val_set))
    error_txy = np.zeros(len(val_set))
    error_q = np.zeros(len(val_set))

    time_results_network = []
    time_results_ransac = []
    # visual_num = 1
    for step, input_dict in enumerate(val_loader):
        val_pose = input_dict['pose']
        batch_size = val_pose.size(0)
        pred_t = np.zeros((batch_size, 3))
        pred_q = np.zeros((batch_size, 4))
        index_list = [0] # index
        start_idx = step * FLAGS.val_batch_size
        end_idx = min((step + 1) * FLAGS.val_batch_size, len(val_set))

        gt_translation[start_idx:end_idx, :] = val_pose[:, :3].numpy() + pose_m
        gt_rotation[start_idx:end_idx, :] = np.asarray([qexp(q) for q in val_pose[:, 3:].numpy()])

        features = input_dict['sinput_F'].to(device, dtype=torch.float32)
        coordinates = input_dict['sinput_C'].to(device)
        pcs_tensor = ME.SparseTensor(features, coordinates)

        coordinates_s8 = input_dict['sinput_s8_C'].to(device)
        features_s8 = input_dict['sinput_s8_F'].to(device, dtype=torch.float32)
        pcs_tensor_s8 = ME.SparseTensor(features_s8, coordinates_s8)

        # inference model and time cost
        start = time.time()
        pred_shift = run_model(model, pcs_tensor, validate=True)
        end = time.time()
        cost_time = (end - start) / FLAGS.val_batch_size
        time_results_network.append(cost_time)

        # gt generation
        ground_truth = pcs_tensor_s8.features_at_coordinates(torch.cat((pred_shift.C[:,0].view(-1,1), pred_shift.C[:,1:]/8), axis=1).float())
        sup_point = ground_truth[:, :3]
        pred_point = sup_point + pred_shift.F
        # 依据下采样点找寻GT
        # 现在情况：输出点是输入点的子集
        for i in range(batch_size):
            # 取出预测的每个batch中的坐标点
            batch_pred_pcs_tensor = pred_shift.coordinates_at(i).float()
            index_list.append(index_list[i] + len(batch_pred_pcs_tensor))

        gt_point = sup_point

        start = time.time()
        for i in range(batch_size):
            batch_pred_t, batch_pred_q = estimate_pose(
                    gt_point[index_list[i]:index_list[i + 1], :] \
                    , pred_point[index_list[i]:index_list[i + 1], :], threshold, device)
            pred_t[i, :] = batch_pred_t
            pred_q[i, :] = batch_pred_q

        end = time.time()
        cost_time = (end - start) / FLAGS.val_batch_size
        time_results_ransac.append(cost_time)

        pred_translation[start_idx:end_idx, :] = pred_t + pose_m
        pred_rotation[start_idx:end_idx, :] = pred_q


        error_t[start_idx:end_idx] = np.asarray([val_translation(p, q) for p, q in
                                                 zip(pred_translation[start_idx:end_idx, :],
                                                     gt_translation[start_idx:end_idx, :])])
        error_txy[start_idx:end_idx] = np.asarray([val_translation(p, q) for p, q in
                                                 zip(pred_translation[start_idx:end_idx, :2],
                                                     gt_translation[start_idx:end_idx, :2])])

        error_q[start_idx:end_idx] = np.asarray([val_rotation(p, q) for p, q in zip(pred_rotation[start_idx:end_idx, :],
                                                                                    gt_rotation[start_idx:end_idx, :])])



        # log_string('ValLoss(m): %f' % float(val_loss))
        log_string('MeanXYZTE(m): %f' % np.mean(error_t[start_idx:end_idx], axis=0))
        log_string('MeanXYTE(m): %f' % np.mean(error_txy[start_idx:end_idx], axis=0))
        log_string('MeanRE(degrees): %f' % np.mean(error_q[start_idx:end_idx], axis=0))
        log_string('MedianTE(m): %f' % np.median(error_t[start_idx:end_idx], axis=0))
        log_string('MedianRE(degrees): %f' % np.median(error_q[start_idx:end_idx], axis=0))

    mean_ATE = np.mean(error_t)
    mean_xyATE = np.mean(error_txy)
    mean_ARE = np.mean(error_q)
    median_ATE = np.median(error_t)
    median_xyATE = np.median(error_txy)
    median_ARE = np.median(error_q)
    mean_time_network = np.mean(time_results_network)
    mean_time_ransac = np.mean(time_results_ransac)

    log_string('Mean Position Error(m): %f' % mean_ATE)
    log_string('Mean XY Position Error(m): %f' % mean_xyATE)
    log_string('Mean Orientation Error(degrees): %f' % mean_ARE)
    log_string('Median Position Error(m): %f' % median_ATE)
    log_string('Median XY Position Error(m): %f' % median_xyATE)
    log_string('Median Orientation Error(degrees): %f' % median_ARE)
    log_string('Mean Network Cost Time(s): %f' % mean_time_network)
    log_string('Mean Ransac Cost Time(s): %f' % mean_time_ransac)

    val_writer.add_scalar('MeanATE', mean_ATE, TOTAL_ITERATIONS)
    val_writer.add_scalar('MeanARE', mean_ARE, TOTAL_ITERATIONS)


    # save error
    error_t_filename = osp.join(FLAGS.log_dir, 'error_t.txt')
    error_q_filename = osp.join(FLAGS.log_dir, 'error_q.txt')
    np.savetxt(error_t_filename, error_t, fmt='%8.7f')
    np.savetxt(error_q_filename, error_q, fmt='%8.7f')

    # trajectory
    fig = plt.figure()
    real_pose = pred_translation - pose_m
    gt_pose = gt_translation - pose_m
    plt.scatter(gt_pose[:, 1], gt_pose[:, 0], s=3, c='black')
    plt.scatter(real_pose[:, 1], real_pose[:, 0], s=3, c='red')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.plot(gt_pose[0, 1], gt_pose[0, 0], 'y*', markersize=10)
    image_filename = os.path.join(os.path.expanduser(FLAGS.log_dir), '{:s}.png'.format('trajectory_' + str(threshold)))
    fig.savefig(image_filename, dpi=200, bbox_inches='tight')

    # translation_distribution
    fig = plt.figure()
    t_num = np.arange(len(error_t))
    plt.scatter(t_num, error_t, s=1, c='red')
    plt.xlabel('Data Num')
    plt.ylabel('Error (m)')
    image_filename = os.path.join(os.path.expanduser(FLAGS.log_dir), '{:s}.png'.format('distribution_t_' + str(threshold)))
    fig.savefig(image_filename, dpi=200, bbox_inches='tight')

    # rotation_distribution
    fig = plt.figure()
    q_num = np.arange(len(error_q))
    plt.scatter(q_num, error_q, s=1, c='blue')
    plt.xlabel('Data Num')
    plt.ylabel('Error (degree)')
    image_filename = os.path.join(os.path.expanduser(FLAGS.log_dir), '{:s}.png'.format('distribution_q_' + str(threshold)))
    fig.savefig(image_filename, dpi=200, bbox_inches='tight')

    # save error and trajectory
    error_t_filename = osp.join(FLAGS.log_dir, 'error_t.txt')
    error_q_filename = osp.join(FLAGS.log_dir, 'error_q.txt')
    pred_q_filename = osp.join(FLAGS.log_dir, 'pred_q.txt')
    pred_t_filename = osp.join(FLAGS.log_dir, 'pred_t.txt')
    gt_t_filename = osp.join(FLAGS.log_dir, 'gt_t.txt')
    np.savetxt(error_t_filename, error_t, fmt='%8.7f')
    np.savetxt(error_q_filename, error_q, fmt='%8.7f')
    np.savetxt(pred_t_filename, real_pose, fmt='%8.7f')
    np.savetxt(pred_q_filename, pred_rotation, fmt='%8.7f')
    np.savetxt(gt_t_filename, gt_pose, fmt='%8.7f')


def run_model(model, x1, validate=False):
    if not validate:
        model.train()
        return model(x1)
    else:
        with torch.no_grad():
            model.eval()
            return model(x1)


if __name__ == "__main__":
    eval()
