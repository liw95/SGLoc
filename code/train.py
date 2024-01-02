# pylint: disable=no-member
import argparse
import os
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import MinkowskiEngine as ME

from data.OxfordVelodyne_datagenerator import Oxford
from data.QEOxfordVelodyne_datagenerator import QEOxford
from data.NCLTVelodyne_datagenerator import NCLT
from models.model import SGLoc
from models.loss import Plane_CriterionCoordinate
from data.base_loader import CollationFunctionFactory
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
cudnn.enabled = True
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0,
                    help='gpu id for network')
parser.add_argument('--batch_size', type=int, default=35,
                    help='Oxford 35 NCLT 30')
parser.add_argument('--val_batch_size', type=int, default=35,
                    help='Batch Size during validating [default: 80]')
parser.add_argument('--max_epoch', type=int, default=51,
                    help='Epoch to run [default: 100]')
parser.add_argument('--init_learning_rate', type=float, default=0.001,
                    help='Initial learning rate [default: 0.001]')
parser.add_argument("--decay_step", type=float, default=1200,
                    help="Oxford: 1200 NCLT: 1000")
parser.add_argument('--optimizer', default='adam',
                    help='adam or momentum [default: adam]')
parser.add_argument('--seed', type=int, default=20, metavar='S',
                    help='random seed (default: 20)')
parser.add_argument('--log_dir', default='log/',
                    help='Log dir [default: log]')
parser.add_argument('--dataset_folder', default='/home/ldq/Codes/SGLoc',
                    help='Our Dataset Folder')
parser.add_argument('--dataset', default='QEOxford',
                    help='Oxford or QEOxford or NCLT')
parser.add_argument('--num_workers', type=int, default=4,
                    help='num workers for dataloader, default:4')
parser.add_argument('--voxel_size', type=float, default=0.2,
                    help='Oxford 0.2 NCLT: 0.25')
parser.add_argument('--skip_val', action='store_true', default=False,
                    help='if skip validation during training, default False')
parser.add_argument('--resume_model', type=str, default='',
                    help='If present, restore checkpoint and resume training')


FLAGS = parser.parse_args()
args = vars(FLAGS)
for (k, v) in args.items():
    print('%s: %s' % (str(k), str(v)))
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)

LOG_FOUT = open(os.path.join(FLAGS.log_dir, 'log.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')
TOTAL_ITERATIONS = 0

# gpu id
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_id)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise ValueError("GPU not found!")

valid_augmentations = []
train_kwargs = dict(data_path=FLAGS.dataset_folder,
                    train=True,
                    valid=False,
                    voxel_size=FLAGS.voxel_size)

valid_kwargs = dict(data_path=FLAGS.dataset_folder,
                    train=False,
                    valid=True,
                    voxel_size=FLAGS.voxel_size)

if FLAGS.dataset == 'Oxford':
    train_set = Oxford(**train_kwargs)
    val_set = Oxford(**valid_kwargs)
    dataset = 'Oxford&QEOxford'
elif FLAGS.dataset == 'QEOxford':
    train_set = QEOxford(**train_kwargs)
    val_set = QEOxford(**valid_kwargs)
    dataset = 'Oxford&QEOxford'
else:
    train_set = NCLT(**train_kwargs)
    val_set = NCLT(**valid_kwargs)
    dataset = 'NCLT'

pose_stats_file = os.path.join(FLAGS.dataset_folder, dataset, FLAGS.dataset + '_pose_stats.txt')
pose_m, pose_s = np.loadtxt(pose_stats_file)

collation_fn = CollationFunctionFactory(collation_type='collate_pair')

train_loader = DataLoader(train_set,
                          batch_size=FLAGS.batch_size,
                          shuffle=True,
                          collate_fn=collation_fn,
                          num_workers=FLAGS.num_workers,
                          pin_memory=True)

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


def train():
    global TOTAL_ITERATIONS
    setup_seed(FLAGS.seed)
    train_writer = SummaryWriter(os.path.join(FLAGS.log_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(FLAGS.log_dir, 'valid'))
    model = SGLoc()
    loss = Plane_CriterionCoordinate()
    model = model.to(device)
    loss = loss.to(device)

    if FLAGS.optimizer == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), FLAGS.init_learning_rate, momentum=0.9)
    elif FLAGS.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), FLAGS.init_learning_rate)
    else:
        optimizer = None
        exit(0)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, FLAGS.decay_step, gamma=0.95)
    if len(FLAGS.resume_model) > 0:
        resume_filename = FLAGS.resume_model
        print("Resuming From ", resume_filename)
        checkpoint = torch.load(resume_filename)
        saved_state_dict = checkpoint['state_dict']
        starting_epoch = checkpoint['epoch'] + 1
        TOTAL_ITERATIONS = starting_epoch * len(train_set)
        model.load_state_dict(saved_state_dict)
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        starting_epoch = 0

    LOG_FOUT.write("\n")
    LOG_FOUT.flush()
    for epoch in range(starting_epoch, FLAGS.max_epoch):
        log_string('**** EPOCH %03d ****' % epoch)
        sys.stdout.flush()
        train_one_epoch(model, train_loader, scheduler, epoch, train_writer, loss, device)
        torch.cuda.empty_cache()


def train_one_epoch(model, train_loader, scheduler, epoch, train_writer, loss, device):
    global TOTAL_ITERATIONS
    for _, input_dict in enumerate(train_loader):
        TOTAL_ITERATIONS += 1
        # input
        features = input_dict['sinput_F'].to(device, dtype=torch.float32)
        coordinates = input_dict['sinput_C'].to(device)
        pcs_tensor = ME.SparseTensor(features, coordinates)
        # 8 times downsampling
        features_s8 = input_dict['sinput_s8_F'].to(device, dtype=torch.float32)
        coordinates_s8 = input_dict['sinput_s8_C'].to(device)
        pcs_tensor_s8  = ME.SparseTensor(features_s8, coordinates_s8)
        # prediction
        scheduler.optimizer.zero_grad()
        pred_shift = run_model(model, pcs_tensor, validate=False)

        ground_truth = pcs_tensor_s8.features_at_coordinates(torch.cat((pred_shift.C[:,0].view(-1,1),
                                                                        pred_shift.C[:,1:]/8), axis=1).float())
        pose_gt = input_dict['pose'].to(device, dtype=torch.float32)
        index_list = [0]
        batch_size = pose_gt.size(0)
        for i in range(batch_size):
            batch_pred_pcs_tensor = pred_shift.coordinates_at(i).float()
            index_list.append(index_list[i] + len(batch_pred_pcs_tensor))

        sup_point = ground_truth[:, :3]
        gt_sup_point = ground_truth[:, 3:6]
        mask = ground_truth[:, 6].view(-1, 1)

        pred_point = sup_point + pred_shift.F

        train_loss = loss(pred_point, gt_sup_point, mask)

        train_loss.backward(train_loss)
        scheduler.optimizer.step()
        scheduler.step()
        log_string('Loss: %f' % train_loss)
        train_writer.add_scalar('Loss', train_loss.cpu().item(), TOTAL_ITERATIONS)

    if epoch % 1 == 0 and epoch>=10:
        if isinstance(model, nn.DataParallel):
            model_to_save = model.module
        else:
            model_to_save = model
        torch.save({
            'epoch': epoch,
            'iter': TOTAL_ITERATIONS,
            'state_dict': model_to_save.state_dict(),
            'scheduler': scheduler.state_dict(),
        },
            FLAGS.log_dir + 'checkpoint_epoch{}.tar'.format(epoch))
        print("Model Saved As " + 'checkpoint_epoch{}.tar'.format(epoch))


def run_model(model, x1, validate=False):
    if not validate:
        model.train()
        return model(x1)
    else:
        with torch.no_grad():
            model.eval()
            return model(x1)


if __name__ == "__main__":
    train()
