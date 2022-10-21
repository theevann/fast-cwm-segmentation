import argparse
import json
import logging
import os, sys

import torch


parser = argparse.ArgumentParser(description="Semantic Segmentation Training")

parser.add_argument('--archi', type=str, default='deeplab',
                    choices=['deeplab', 'swiftnet', 'custom'],
                    help='Architecture name (default: deeplab)')
parser.add_argument('--archi-name', type=str,
                     help='Architecture name for folder')
parser.add_argument('--workers', type=int, default=6,
                    help='dataloader threads')
parser.add_argument('--sync-bn', type=bool, default=None,
                    help='whether to use sync bn (default: auto)')

# Deeplab specifics
parser.add_argument('--backbone', type=str, default='resnet101',
                    choices=['resnet101', 'resnet50',
                             'resnet18', 'mobilenet'],
                    help='backbone for deeplab (default: resnet101)')
parser.add_argument('--out-stride', type=int, default=16,
                    help='network output stride for deeplab (default: 16)')
parser.add_argument('--freeze-bn', type=bool, default=False,
                    help='whether to freeze bn parameters (default: False)')
parser.add_argument('--no-image-lvl-feat', action='store_true', default=False,
                    help='Do not use image level features in DeepLab ASPP (default: False)')

# DB
parser.add_argument('--db-root', type=str, default=os.getenv("DB_ROOT"),
                    help='Root directory for db')
parser.add_argument('--frame-numbers', type=int, nargs='+', default=[19],
                    help='frames to use for training (default: [19])')
parser.add_argument('--crop-size', type=int, nargs='+', default=[550],
                    help='crop image size (default: [550])')


# Training hyper params
parser.add_argument('--epochs', type=int, default=None,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=4,
                    help='batch size for training (default: 4)')
parser.add_argument('--val-batch-size', type=int, default=4,
                    help='batch size for validation (default: 4)')

# Optimizer params
parser.add_argument('--optim', type=str, default='sgd',
                    choices=['sgd', 'adam'],
                    help='optimizer (default: sgd)')
parser.add_argument('--lr', type=float,
                    help='learning rate')
parser.add_argument('--weight-decay', type=float, default=5e-4,
                    help='w-decay (default: 5e-4)')

## SGD
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--nesterov', action='store_true', default=False,
                    help='whether use nesterov (default: False)')

# Scheduler
parser.add_argument('--scheduler', type=str, default='poly',
                    choices=['cos', 'poly'],
                    help='lr scheduler mode: (default: poly)')
parser.add_argument('--poly-power', type=float, default=0.9,
                    help='poly scheduler power (default: 0.9)')
parser.add_argument('--cos-eta-min', type=float, default=1e-6,
                    help='cos scheduler min lr (default: 1e-6)')

# Cuda, seed and logging
parser.add_argument('--no-cuda', action='store_true',
                    default=False, help='disables CUDA training')
parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0],
                    help='use which gpu to train (default=0)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-level', type=str, default="INFO",
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                    help='log level (default: INFO)')

# Checking point
parser.add_argument('--resume', type=str, default=None,
                    help='Path to the checkpoint file')
parser.add_argument('--checkname', type=str, default=None,
                    help='Set the subdirectory name')
parser.add_argument('--save-dir', type=str, default='run',
                    help='Set the directory for saving runs')
parser.add_argument('--xp-folder', type=str, default=None)
parser.add_argument('--save-every-epoch', action="store_true", default=False)

# Finetuning pre-trained models
parser.add_argument('--ft', action='store_true', default=False,
                    help='finetuning on a different dataset')

# Evaluation option
parser.add_argument('--eval-interval', type=int, default=2,
                    help='evaluation interval (default: 2)')
parser.add_argument('--eval-only', type=bool, default=False,
                    help='only do one validation (default: False)')


# Additional options for slimmed CWM
parser.add_argument('--mix', type=int, nargs='+', default=[0],
                    help='which mix to train with (default=[0])')
parser.add_argument('--masks', type=int, nargs='+', default=[0,1],
                    help='which masks to train with (default=[0,1])')
parser.add_argument('--generator', type=str, default='bistep')
parser.add_argument('--width-mult', type=float, default=1,
                    help='Width multiplier for slimmable (default=1)')


def load_args(config_file):
    args = parser.parse_args("")
    args_json = json.load(config_file.open('r'))
    for k, v in args_json.items(): setattr(args, k, v)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.sync_bn is None:
        args.sync_bn = (args.cuda and len(args.gpu_ids) > 1)

    args.db = {
        'root': args.db_root,
        'frame_numbers': args.frame_numbers,
        'crop_size': args.crop_size,
    }

    logging.basicConfig(stream=sys.stdout, format='%(message)s')
    args.logger = logging.getLogger('train_code')
    args.logger.setLevel(getattr(logging, args.log_level))

    return args
