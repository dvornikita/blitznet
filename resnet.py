import logging
import os

import tensorflow as tf

import resnet_v1
import resnet_utils

from resnet_v1 import bottleneck, bottleneck_skip
from utils import print_variables
import config
from config import args, MEAN_COLOR
from paths import INIT_WEIGHTS_DIR


log = logging.getLogger()
slim = tf.contrib.slim

CKPT_50 = os.path.join(INIT_WEIGHTS_DIR, 'resnet50_full.ckpt')
DEFAULT_SCOPE_50 = 'resnet_v1_50'
DEFAULT_SSD_SCOPE = 'ssd'


class ResNet(object):
    def __init__(self, config, training, weight_decay=0.0005, depth=50,
                 reuse=False):
        self.config = config
        self.weight_decay = weight_decay
        self.layers = []
        self.reuse = reuse
        self.training = training
        self.layers = self.config['layers']
        if depth == 50:
            self.num_block3 = 5
            self.scope = DEFAULT_SCOPE_50
            self.ckpt = CKPT_50
        else:
            raise ValueError

    def create_trunk(self, images):
        red, green, blue = tf.split(images*255, 3, axis=3)
        images = tf.concat([blue, green, red], 3) - MEAN_COLOR

        with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training=self.training,
                                                       weight_decay=self.weight_decay,
                                                       batch_norm_decay=args.bn_decay)):
            blocks = [
                resnet_utils.Block(
                    'block1', bottleneck, [(256, 64, 1)] * 3),
                resnet_utils.Block(
                    'block2', bottleneck, [(512, 128, 2)] + [(512, 128, 1)] * 3),
                resnet_utils.Block(
                    'block3', bottleneck, [(1024, 256, 2)] + [(1024, 256, 1)] * self.num_block3),
                resnet_utils.Block(
                    'block4', bottleneck, [(2048, 512, 2)] + [(2048, 512, 1)] * 2)
            ]

            net, endpoints = resnet_v1.resnet_v1(images, blocks,
                                                 global_pool=False,
                                                 reuse=self.reuse,
                                                 scope=self.scope)
            self.outputs = endpoints
        self.add_extra_layers(net)

    def vgg_arg_scope(self):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(self.weight_decay),
                            biases_initializer=tf.zeros_initializer(),
                            padding='SAME') as arg_sc:
            return arg_sc

    def add_extra_layers(self, net):
        with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training=self.training,
                                                       weight_decay=self.weight_decay,
                                                       batch_norm_decay=args.bn_decay)):
            block_depth = 2
            num_fm = 2048
            blocks = [
                resnet_utils.Block(
                    'block5', bottleneck, [(num_fm, num_fm//4, 2)] + [(num_fm, num_fm//4, 1)] * (block_depth-1)),
                resnet_utils.Block(
                    'block6', bottleneck, [(num_fm, num_fm//4, 2)] + [(num_fm, num_fm//4, 1)] * (block_depth-1)),
                resnet_utils.Block(
                    'block7', bottleneck, [(num_fm, num_fm//4, 2)] + [(num_fm, num_fm//4, 1)] * (block_depth-1)),
            ]
            if args.image_size == 512:
                blocks += [
                    resnet_utils.Block(
                        'block8', bottleneck, [(num_fm, num_fm//4, 2)] + [(num_fm, num_fm//4, 1)] * (block_depth-1)),
                ]

            net, endpoints = resnet_v1.resnet_v1(net, blocks,
                                                 global_pool=False,
                                                 include_root_block=False,
                                                 reuse=self.reuse,
                                                 scope=DEFAULT_SSD_SCOPE)
            self.outputs.update(endpoints)
            with tf.variable_scope(DEFAULT_SSD_SCOPE+"_back", reuse=self.reuse):
                end_points_collection = "reverse_ssd_end_points"
                with slim.arg_scope([slim.conv2d, bottleneck_skip],
                                    outputs_collections=end_points_collection):
                    top_fm = args.top_fm
                    int_fm = top_fm//4
                    if args.image_size == 512:
                        # as long as the number of pooling layers is bigger due to
                        # the higher resolution, an extra layer is appended
                        net = bottleneck_skip(net, self.outputs[DEFAULT_SSD_SCOPE+'/block7'],
                                              top_fm, int_fm, scope='block_rev7')

                    net = bottleneck_skip(net, self.outputs[DEFAULT_SSD_SCOPE+'/block6'],
                                          top_fm, int_fm, scope='block_rev6')
                    net = bottleneck_skip(net, self.outputs[DEFAULT_SSD_SCOPE+'/block5'],
                                          top_fm, int_fm, scope='block_rev5')
                    net = bottleneck_skip(net, self.outputs[self.scope+'/block4'],
                                          top_fm, int_fm, scope='block_rev4')
                    net = bottleneck_skip(net, self.outputs[self.scope+'/block3'],
                                          top_fm, int_fm, scope='block_rev3')
                    net = bottleneck_skip(net, self.outputs[self.scope+'/block2'],
                                          top_fm, int_fm, scope='block_rev2')
                    if args.x4:
                        # To provide stride 4 we add one more layer with upsampling
                        net = bottleneck_skip(net, self.outputs[self.scope+'/block1'],
                                              top_fm, int_fm, scope='block_rev1')
                endpoints = slim.utils.convert_collection_to_dict(end_points_collection)
            self.outputs.update(endpoints)

            # Creating an output of spatial resolution 1x1 with conventional name 'pool6'
            if args.image_size == 512:
                self.outputs[DEFAULT_SSD_SCOPE+'/pool6'] =\
                        tf.reduce_mean(self.outputs['ssd_back/block_rev7/shortcut'],
                                       [1, 2], name='pool6', keep_dims=True)
            else:
                self.outputs[DEFAULT_SSD_SCOPE+'/pool6'] =\
                        tf.reduce_mean(self.outputs['ssd_back/block_rev6/shortcut'],
                                       [1, 2], name='pool6', keep_dims=True)

    def create_multibox_head(self, num_classes):
        """Creates outputs for classification and localization of all candidate bboxes"""
        locations = []
        confidences = []
        with tf.variable_scope(DEFAULT_SSD_SCOPE, reuse=self.reuse) as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope(self.vgg_arg_scope()):
                with slim.arg_scope([slim.conv2d], outputs_collections=end_points_collection,
                                    weights_initializer=slim.variance_scaling_initializer(factor=0.1),
                                    activation_fn=None):
                    for i, layer_name in enumerate(self.layers):

                        if i > 0 and args.head == 'shared':
                            sc.reuse_variables()
                        if args.head == 'shared':
                            scope_suffix = ''
                        elif args.head == 'nonshared':
                            scope_suffix = '/'+layer_name
                        else:
                            raise ValueError
                        src_layer = self.outputs[layer_name]
                        shape = src_layer.get_shape()
                        wh = shape[1] * shape[2]
                        batch_size = shape[0]
                        num_priors = len(self.config['aspect_ratios'][i])*2 + 2

                        loc = slim.conv2d(src_layer, num_priors * 4,
                                          [args.det_kernel, args.det_kernel],
                                          scope='location'+scope_suffix)
                        loc_sh = tf.stack([batch_size, wh * num_priors, 4])
                        locations.append(tf.reshape(loc, loc_sh))
                        tf.summary.histogram("location/"+layer_name, locations[-1])

                        conf = slim.conv2d(src_layer, num_priors * num_classes,
                                           [args.det_kernel, args.det_kernel],
                                           scope='confidence'+scope_suffix)
                        conf_sh = tf.stack([batch_size, wh * num_priors, num_classes])
                        confidences.append(tf.reshape(conf, conf_sh))
                        tf.summary.histogram("confidence/"+layer_name, confidences[-1])

                    ssd_end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                    self.outputs.update(ssd_end_points)
        all_confidences = tf.concat(confidences, 1)
        all_locations = tf.concat(locations, 1)
        self.outputs['location'] = all_locations
        self.outputs['confidence'] = all_confidences
        return all_confidences, all_locations

    def create_segmentation_head(self, num_classes):
        """segmentation of map with stride 8 or 4, if --x4 flag is active"""
        with tf.variable_scope(DEFAULT_SSD_SCOPE) as sc:
            with slim.arg_scope([slim.conv2d],
                                kernel_size=args.seg_filter_size,
                                weights_regularizer=slim.l2_regularizer(self.weight_decay),
                                biases_initializer=tf.zeros_initializer()):

                seg_materials = []
                seg_size = self.config['fm_sizes'][0]
                for i in range(len(self.layers)):
                    target_layer = self.outputs[self.layers[i]]
                    seg = slim.conv2d(target_layer, args.n_base_channels)
                    seg = tf.image.resize_nearest_neighbor(seg, [seg_size, seg_size])
                    seg_materials.append(seg)
                seg_materials = tf.concat(seg_materials, -1)
                seg_logits = slim.conv2d(seg_materials, num_classes,
                                        kernel_size=3, activation_fn=None)
                self.outputs['segmentation'] = seg_logits
                return self.outputs['segmentation']

    def get_imagenet_init(self, opt):
        """optimizer is useful to extract slots corresponding to Adam or Momentum
        and exclude them from checkpoint assigning"""
        variables = slim.get_variables_to_restore(include=[self.scope])
        slots = set()
        for v in tf.trainable_variables():
            for s in opt.get_slot_names():
                slot = opt.get_slot(v, s)
                if slot is not None:
                    slots.add(slot)
        variables = list(set(variables) - slots)
        return slim.assign_from_checkpoint(self.ckpt, variables) + (variables, )

