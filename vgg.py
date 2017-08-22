import os
import tensorflow as tf
from config import config, MEAN_COLOR
from paths import INIT_WEIGHTS_DIR

slim = tf.contrib.slim

ATROUS_CKPT = os.path.join(INIT_WEIGHTS_DIR, 'atrous.ckpt')
DEFAULT_SCOPE = 'vgg_16'
DEFAULT_SSD_SCOPE = 'ssd'


class VGG(object):
    def __init__(self, config, training=True, weight_decay=0.0005, depth=16, scope=DEFAULT_SCOPE, reuse=False):
        self.scope = scope
        self.config = config
        self.weight_decay = weight_decay
        self.layers = []
        self.reuse = reuse
        for name in self.config['layers']:
            if 'conv' in name:
                self.layers.append('%s/%s/%s' % (self.scope, name[:5], name))
            else:
                self.layers.append('%s/%s' % (self.scope, name))

    def vgg_arg_scope(self):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(self.weight_decay),
                            biases_initializer=tf.zeros_initializer(),
                            padding='SAME') as arg_sc:
            return arg_sc

    def create_multibox_head(self, num_classes):
        locations = []
        confidences = []

        with tf.variable_scope(DEFAULT_SSD_SCOPE, DEFAULT_SSD_SCOPE, [self.outputs[k] for k in self.layers], reuse=self.reuse) as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope(self.vgg_arg_scope()):
                with slim.arg_scope([slim.conv2d], outputs_collections=end_points_collection, activation_fn=None):
                    scale_mult = tf.get_variable("conv4_3_scale_mult", (512,), tf.float32, tf.constant_initializer(20.0))
                    tf.summary.histogram("scale_mult", scale_mult)
                    lname = '%s/conv4/conv4_3' % self.scope
                    self.outputs[lname] = tf.nn.l2_normalize(self.outputs[lname], (1, 2), name='conv4_3_l2_normalization')*tf.reshape(scale_mult, (1, 1, 1, 512))
                    for i, layer_name in enumerate(self.layers):
                        src_layer = self.outputs[layer_name]
                        shape = src_layer.get_shape()
                        wh = shape[1] * shape[2]
                        batch_size = shape[0]
                        num_priors = len(self.config['aspect_ratios'][i])*2 + 2

                        loc = slim.conv2d(src_layer, num_priors * 4, [3, 3],
                                        scope=layer_name+'/location')
                        loc_sh = tf.stack([batch_size, wh * num_priors, 4])
                        locations.append(tf.reshape(loc, loc_sh))
                        tf.summary.histogram("location/"+layer_name, locations[-1])

                        conf = slim.conv2d(src_layer, num_priors * num_classes, [3, 3],
                                        scope=layer_name+'/confidence')
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

    def create_trunk(self, images):
        # Convert RGB to BGR
        red, green, blue = tf.split(images*255, 3, axis=3)
        inputs = tf.concat([blue, green, red], 3) - MEAN_COLOR
        with slim.arg_scope(self.vgg_arg_scope()):
            with tf.variable_scope(self.scope, DEFAULT_SCOPE, [inputs], reuse=self.reuse) as sc:
                end_points_collection = sc.name + '_end_points'
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    outputs_collections=end_points_collection):
                    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                    net = slim.max_pool2d(net, [2, 2], scope='pool3')
                    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                    net = slim.max_pool2d(net, [2, 2], scope='pool4')
                    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')

                    net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5', padding='SAME')
                    net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='fc6')
                    net = slim.conv2d(net, 1024, [1, 1], scope='fc7')

                    net = slim.stack(net, slim.conv2d, [(256, 1, 1), (512, 3, 2)], scope='conv6')
                    net = slim.stack(net, slim.conv2d, [(128, 1, 1), (256, 3, 2)], scope='conv7')
                    with slim.arg_scope([slim.conv2d], padding="VALID"):
                        net = slim.stack(net, slim.conv2d, [(128, 1), (256, 3)], scope='conv8')
                        net = slim.stack(net, slim.conv2d, [(128, 1), (256, 3)], scope='conv9')

                    self.outputs = slim.utils.convert_collection_to_dict(end_points_collection)

    def get_imagenet_init(self, opt):
        # optimizer is useful to extract slots corresponding to Adam or Momentum
        # and exclude them from checkpoint assigning
        vgg_names = (['%s/conv%i' % (self.scope, i) for i in range(1, 6)] +
                     ['%s/fc%i' % (self.scope, i) for i in range(6, 8)])
        variables = slim.get_variables_to_restore(include=vgg_names)
        slots = set()
        for v in tf.trainable_variables():
            for s in opt.get_slot_names():
                slot = opt.get_slot(v, s)
                if slot is not None:
                    slots.add(slot)
        variables = list(set(variables) - slots)
        ckpt = ATROUS_CKPT
        return slim.assign_from_checkpoint(ckpt, variables) + (variables, )
