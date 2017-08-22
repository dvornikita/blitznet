import sys
import os
import argparse
from paths import *

# Mean color to subtract before propagating an image through a DNN
MEAN_COLOR = [103.062623801, 115.902882574, 123.151630838]

parser = argparse.ArgumentParser(description='Train or eval SSD model with goodies.')

# The name of your experiment
parser.add_argument("--run_name", type=str, required=True)

# The number of checkpoint (in thousands) you want to restore from
parser.add_argument("--ckpt", default=0, type=int)

# The dataset you want to train/test the model on
parser.add_argument("--dataset", default='voc07', choices=['voc07', 'voc12', 'voc07+12',
                                                           'coco', 'voc07+12-segfull',
                                                           'voc07+12-segmentation',
                                                           'coco-seg'])

# The split of the dataset you want to train/test on
parser.add_argument("--split", default='train', choices=['train', 'test', 'val', 'trainval',
                                                         'train-segmentation', 'val-segmentation',
                                                         'train-segmentation-original',
                                                         'valminusminival2014', 'minival2014',
                                                         'test-dev2015', 'test2015'])

# The network you use as a base network (backbone)
parser.add_argument("--trunk", default='resnet50', choices=['resnet50', 'vgg16'])

# Either the last layer has a stride of 4 of of 8, if True an extra layer is appended
parser.add_argument("--x4", default=False, action='store_true')

# Which image size to chose for training
parser.add_argument("--image_size", default=300, type=int)

# If True, shares the weights for classifiers of bboxes on each scale
parser.add_argument("--head", default='nonshared', choices=['shared', 'nonshared'])

# Sampling method for deep features resizing
parser.add_argument("--resize", default='bilinear', choices=['bilinear', 'nearest'])

# The number of feature maps in the layers appended to a base network
parser.add_argument("--top_fm", default=512, type=int)

# The size of conv kernel in classification/localization mapping for bboxes
parser.add_argument("--det_kernel", default=3, type=int)

# TRAINING FLAGS
parser.add_argument("--max_iterations", default=1000000, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--weight_decay", default=5e-5, type=float)
parser.add_argument("--bn_decay", default=0.9, type=float)
parser.add_argument("--learning_rate", default=1e-4, type=float)

# For training with warmup, chose the number of steps
parser.add_argument("--warmup_step", default=0, type=int)

#For training with warmup, chose the starting learning rate
parser.add_argument("--warmup_lr", default=1e-5, type=float)

# Optimizer of choice
parser.add_argument("--optimizer", default='adam', choices=['adam', 'nesterov'])

# To what ratio of images apply zoomout data augmentation
parser.add_argument("--zoomout_prob", default=0.5, type=float)

# A list of steps where after each a learning rate is multiplied by 1e-1
parser.add_argument("--lr_decay", default=[], nargs='+', type=int)

# Random initialization of a base network
parser.add_argument("--random_trunk_init", default=False, action='store_true')

# SEGMENTATION/DETECTION FLAGS
# if you want a net to perform detection
parser.add_argument("--detect", default=False, action='store_true')

# if you want a network to perform segmentation
parser.add_argument("--segment", default=False, action='store_true')

# Nope
parser.add_argument("--no_seg_gt", default=False, action='store_true')

# The size of intermediate representations before concatenating and segmenting
parser.add_argument("--n_base_channels", default=64, type=int)

# The size of the conv filter used to map feature maps to intermediate representations before segmentation
parser.add_argument("--seg_filter_size", default=1, type=int, choices=[1, 3])

# EVALUATION FLAGS
# Automatic evaluation of several checkpoints
parser.add_argument("--batch_eval", default=False, action='store_true')

# number of checkpoint in thousands you want to start the evaluation from
parser.add_argument("--min_ckpt", default=0, type=int)

# a step between checkpoints to evaluate in thousands
parser.add_argument("--step", default=1, type=int)

# How many top scoring bboxes per category are passed to nms
parser.add_argument("--top_k_nms", default=400, type=int)

# How many top scoring bboxes per category are left after nms
parser.add_argument("--top_k_after_nms", default=50, type=int)

# How many top scoring bboxes in total are left after nms for an image
parser.add_argument("--top_k_post_nms", default=200, type=int)

# The threshold of confidence above which a bboxes is considered as a class example
parser.add_argument("--conf_thresh", default=0.01, type=float)

# IoU threshold for nms
parser.add_argument("--nms_thresh", default=0.45, type=float)

# IoU threshold positive criteria in PASCAL VOC challenge
parser.add_argument("--voc_iou_thresh", default=0.50, type=float)

# Filter candidate boxes by thresholding the score.
# Needed to make clean final detection results.
parser.add_argument("--eval_min_conf", default=0.0, type=float)

# First n processed images will be saved with regressed bboxes/masks drawn
parser.add_argument("--save_first_n", default=0, type=int)

args = parser.parse_args()
train_dir = os.path.join(CKPT_ROOT, args.run_name)

# Configurations for data augmentation
data_augmentation_config = {
    'X_out': 4,
    'brightness_prob': 0.5,
    'brightness_delta': 0.125,
    'contrast_prob': 0.5,
    'contrast_delta': 0.5,
    'hue_prob': 0.5,
    'hue_delta': 0.07,
    'saturation_prob': 0.5,
    'saturation_delta': 0.5,
    'sample_jaccards': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
    'flip_prob': 0.5,
    'crop_max_tries': 50,
    'zoomout_color': [x/255.0 for x in reversed(MEAN_COLOR)],
}

config_vgg = {
    'image_size': 300,
    'smallest_scale': 0.1,
    'min_scale': 0.2,
    'max_scale': 0.9,
    'layers': ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2'],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'prior_variance': [0.1, 0.1, 0.2, 0.2],
    'train_augmentation': data_augmentation_config,
    'fm_sizes': [37, 18, 9, 5, 3, 1],
}

evaluation_logfile = '1evaluations.txt'
normAP_constant = 400


config_resnet_ssd512_x4 = {'image_size': 512,
                           'smallest_scale': 0.02,
                           'min_scale': 0.08,
                           'max_scale': 0.95,
                           'layers': ['ssd_back/block_rev1', 'ssd_back/block_rev2', 'ssd_back/block_rev3', 'ssd_back/block_rev4', 'ssd_back/block_rev5', 'ssd_back/block_rev6', 'ssd_back/block_rev7', 'ssd/pool6'],
                           'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
                           'train_augmentation': data_augmentation_config,
                           'prior_variance': [0.1, 0.1, 0.2, 0.2],
                           'fm_sizes': [128, 64, 32, 16, 8, 4, 2, 1],
}

config_resnet_ssd512_nox4 = {'image_size': 512,
                             'smallest_scale': 0.04,
                             'min_scale': 0.1,
                             'max_scale': 0.95,
                             'layers': ['ssd_back/block_rev2', 'ssd_back/block_rev3', 'ssd_back/block_rev4', 'ssd_back/block_rev5', 'ssd_back/block_rev6', 'ssd_back/block_rev7', 'ssd/pool6'],
                             'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
                             'train_augmentation': data_augmentation_config,
                             'prior_variance': [0.1, 0.1, 0.2, 0.2],
                             'fm_sizes': [64, 32, 16, 8, 4, 2, 1],
}

config_resnet_nox4 = {'image_size': 300,
                      'smallest_scale': 0.1,
                      'min_scale': 0.2,
                      'max_scale': 0.95,
                      'layers': ['ssd_back/block_rev2', 'ssd_back/block_rev3', 'ssd_back/block_rev4', 'ssd_back/block_rev5', 'ssd_back/block_rev6', 'ssd/pool6'],
                      'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
                      'train_augmentation': data_augmentation_config,
                      'prior_variance': [0.1, 0.1, 0.2, 0.2],
                      'fm_sizes': [38, 19, 10, 5, 3, 1],
}

config_resnet_x4 = {'image_size': 300,
                    'smallest_scale': 0.04,
                    'min_scale': 0.1,
                    'max_scale': 0.95,
                    'layers': [ 'ssd_back/block_rev1', 'ssd_back/block_rev2', 'ssd_back/block_rev3', 'ssd_back/block_rev4', 'ssd_back/block_rev5', 'ssd_back/block_rev6', 'ssd/pool6'],
                    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
                    'train_augmentation': data_augmentation_config,
                    'prior_variance': [0.1, 0.1, 0.2, 0.2],
                    'fm_sizes': [75, 38, 19, 10, 5, 3, 1],
}


if args.trunk == 'resnet50' and args.x4 and args.image_size == 300:
    config = config_resnet_x4
if args.trunk == 'resnet50' and args.x4 and args.image_size == 512:
    config = config_resnet_ssd512_x4
if args.trunk == 'vgg16' and args.x4:
    raise NotImplementedError
if args.trunk in ['resnet50', 'resnet101'] and not args.x4 and args.image_size == 300:
    config = config_resnet_nox4
if args.trunk in ['resnet50', 'resnet101'] and not args.x4 and args.image_size == 512:
    config = config_resnet_ssd512_nox4
if args.trunk == 'vgg16' and not args.x4:
    config = config_vgg


def get_logging_config(run):
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s]: %(message)s'
            },
            'short': {
                'format': '[%(levelname)s]: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'formatter': 'short',
                'class': 'logging.StreamHandler',
            },
            'file': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': LOGS+run+'.log'
            },
        },
        'loggers': {
            '': {
                'handlers': ['default', 'file'],
                'level': 'DEBUG',
                'propagate': True
            },
        }
    }
