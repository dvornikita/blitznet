import numpy as np
import logging
from utils_tf import batch_iou_tf, encode_bboxes_tf

import tensorflow as tf

log = logging.getLogger()


class PriorBoxGrid():
    """Class that creates a greed of candidate bounding boxes
    and matches them with the gt"""
    def __init__(self, config, sess=None, iou_threshold=0.5):
        self.sess = sess
        self.config = config
        self.fm_sizes = config['fm_sizes']
        self.iou_threshold = iou_threshold
        self.tiling = []
        self.min_scale_vars = []
        self.max_scale_vars = []
        self.ar_vars = []
        self.tile_tf()

    def _init_variable(self, name, val):
        """Creates a variable/constant in tf"""
        # return tf.get_variable(name, (), dtype=tf.float32, trainable=False,
        #                        initializer=tf.constant_initializer(val))
        return tf.constant(name=name, shape=(), dtype=tf.float32, value=val)

    def tile_tf(self):
        def initialize_aspect_ratios(ars):
            assert len(self.ar_vars) == 0
            with tf.variable_scope('aspect_ratios'):
                for layer_n, layer in enumerate(self.config['layers']):
                    with tf.variable_scope(layer):
                        layer_ar = []
                        for i in range(len(ars[layer_n])):
                            ar = ars[layer_n][i]
                            ar_var_vert = self._init_variable("ar_%i_vert" % i, ar)
                            ar_var_horiz = self._init_variable("ar_%i_horiz" % i, 1.0/ar)
                            layer_ar.extend([ar_var_vert, ar_var_horiz])
                        self.ar_vars.append(layer_ar)

        def get_scales(nb_fmaps=len(self.config['layers']),
                       min_scale=self.config['min_scale'],
                       max_scale=self.config['max_scale']):
            step = (max_scale - min_scale) / (nb_fmaps - 2)
            min_sizes, max_sizes = [], []
            for ratio in np.arange(min_scale, max_scale + 0.01, step):
                min_sizes.append(ratio)
                max_sizes.append(ratio + step)
            min_sizes = np.array([self.config['smallest_scale']] + min_sizes)
            max_sizes = np.array([min_scale] + max_sizes)

            with tf.variable_scope('scales'):
                for i, layer in enumerate(self.config['layers']):
                    with tf.variable_scope(layer):
                        self.min_scale_vars.append(self._init_variable('min', min_sizes[i]))
                        self.max_scale_vars.append(self._init_variable('max', max_sizes[i]))
            min_scales_tf = tf.stack(self.min_scale_vars, 0, name='min_sca1es')
            max_scales_tf = tf.stack(self.max_scale_vars, 0, name='max_sca1es')
            return (min_scales_tf, max_scales_tf)

        def adjust_for_aspect_ratio(bbox_set, ar):
            """gets set bboxes of aspect ratio 1 and transforms
            them in order to get rectangular bboxes"""
            new_wh = tf.stack([bbox_set[..., 2] * tf.sqrt(ar), bbox_set[..., 3] / tf.sqrt(ar)], -1)
            return tf.concat([bbox_set[..., 0:2], new_wh], -1)

        def generate_boxes(fm_side, scale, aspect_ratios=[]):
            """generates a regular grid fm_size * fm_size of bboxes
            corresponding to the current scale"""
            stride_space = tf.linspace(0.5 / fm_side, 1 - 0.5 / fm_side, fm_side)
            yv, xv = tf.meshgrid(stride_space, stride_space, indexing='ij')
            h_s, w_s = tf.zeros_like(xv) + scale, tf.zeros_like(xv) + scale
            xywh_space = tf.stack([xv, yv, w_s, h_s], 2)
            bbox_set = tf.reshape(xywh_space, [fm_side, fm_side, 1, 4])
            priors = [bbox_set]
            for aspect_ratio in aspect_ratios:
                # using a reference grid of square bboxes generates asymmetric bboxes
                priors.append(adjust_for_aspect_ratio(bbox_set, aspect_ratio))
            return priors

        with tf.variable_scope('tiling'):
            initialize_aspect_ratios(self.config['aspect_ratios'])
            scales, extra_scales = get_scales()

            for i in range(len(self.config['layers'])):
                layer_boxes = generate_boxes(self.fm_sizes[i], scales[i], self.ar_vars[i])
                # introduces an extra scale for each layer
                new_scale = tf.sqrt(tf.cast(scales[i]*extra_scales[i], tf.float32))
                layer_boxes.extend(generate_boxes(self.fm_sizes[i], new_scale))
                layer_boxes = tf.concat(layer_boxes, 2)
                self.tiling.append(tf.reshape(layer_boxes, [-1, 4]))

            self.tiling = tf.concat(self.tiling, 0)
            new_xy = (self.tiling[:, :2] - self.tiling[:, 2:] / 2)
            self.tiling = tf.concat([new_xy, self.tiling[:, 2:]], 1)

        log.debug('number of anchor boxes: %i', self.tiling.get_shape().as_list()[0])

    def get_tiling_params(self):
        return (self.min_scale_vars, self.max_scale_vars, self.ar_vars)

    def encode_gt_tf(self, gt_boxes, gt_cats):
        """Matching of candidate bboxes with gt in tensorflow
        Attention: very painful"""
        n_pos = tf.cast(tf.shape(gt_cats)[0], tf.int32)
        empty_gt = tf.equal(n_pos, 0)

        gt_cats = tf.cast(gt_cats, tf.int32)
        # ugliest hack ever, but I didn't find any other way to avoid
        # reduction of zero-length arrays below if we don't have GT
        gt_boxes = tf.cond(empty_gt, lambda: tf.zeros((1, 4)), lambda: gt_boxes)
        gt_cats = tf.cond(empty_gt, lambda: tf.zeros((1, ), dtype=tf.int32), lambda: gt_cats)

        # source for tiling if nothing matches
        positive_vec0 = tf.zeros(self.tiling.get_shape()[0], dtype=tf.bool)
        cats_vec0 = tf.zeros(self.tiling.get_shape()[0], dtype=tf.int32)
        gt_bboxes_vec0 = tf.ones((tf.shape(self.tiling)[0], 4), dtype=tf.float32)

        iou = batch_iou_tf(self.tiling, gt_boxes)

        # For each prior box we put in correspondance the closest gt box
        closest_gt_inds = tf.cast(tf.argmax(iou, axis=1), tf.int32)
        # hacky solution to use scatter_nd
        # you should just write down intermediate matrices to grok it
        embed_inds = tf.stack([tf.range(tf.size(closest_gt_inds)), closest_gt_inds], 1)
        embed_values = tf.ones_like(closest_gt_inds)
        pos_matches_gt = tf.scatter_nd(embed_inds, embed_values, tf.shape(iou))
        # remove weak matches
        pos_matches_gt = pos_matches_gt * tf.cast(iou >= self.iou_threshold, tf.int32)

        # For each gt box we put in correspondance the closest prior box
        closest_prior_inds = tf.cast(tf.argmax(iou, axis=0), tf.int32)
        embed_inds = tf.stack([closest_prior_inds, tf.range(tf.size(closest_prior_inds))], 1)
        embed_values = tf.ones_like(closest_prior_inds)
        pos_matches_prior = tf.scatter_nd(embed_inds, embed_values, tf.shape(iou))

        # find out what matches on previous step
        # to clean "double" matches (the same prior is the closest for multiple gt)
        numb_of_matches = tf.reduce_sum(pos_matches_prior, axis=1)
        nonempty_priors_mask = numb_of_matches > 0
        emb_x = tf.boolean_mask(tf.range(tf.size(nonempty_priors_mask), dtype=tf.int32),
                                nonempty_priors_mask)
        # this argmax imposes uniqueness because given a string
        # [0 ... 0 1 0 ... 0 1 0 ... 0]
        # it returns the index of the first "1" which will be
        # then projected back by scatter_nd
        emb_y = tf.boolean_mask(tf.cast(tf.argmax(pos_matches_prior, 1), tf.int32),
                                nonempty_priors_mask)
        emb_inds = tf.stack([emb_x, emb_y], 1)
        emb_values = tf.ones_like(emb_y)
        pos_matches_unique = tf.scatter_nd(emb_inds, emb_values, tf.shape(pos_matches_prior))

        # we need to replace lines where we double matched with clean lines
        # having only one non-zero element. Actually what we do, we just replace all
        # lines even with simple matches. But it should not change anything.
        cond = tf.reduce_sum(pos_matches_prior, axis=1) > 0
        pos_matches = tf.where(cond, pos_matches_unique, pos_matches_gt)

        # binary mask of tiling (if a box matches to anything)
        positive_vec = tf.cast(tf.reduce_sum(pos_matches, axis=1), tf.bool)
        # corresponding indices of GT for each positive match
        gt_matches = tf.argmax(pos_matches, axis=1)
        # corresponding classes for each positive match
        cats_vec = tf.gather(gt_cats, gt_matches)
        # GT coordinates for each positive match
        gt_bboxes_vec = tf.gather(gt_boxes, gt_matches)

        # TF creates weird stuff if nothing matches at all
        # we need to cleanup that stuff
        positive_vec = tf.cond(empty_gt, lambda: positive_vec0, lambda: positive_vec)
        cats_vec = tf.cond(empty_gt, lambda: cats_vec0, lambda: cats_vec)
        cats_vec = tf.where(positive_vec, cats_vec, cats_vec0)
        gt_bboxes_vec = tf.cond(empty_gt, lambda: gt_bboxes_vec0, lambda: gt_bboxes_vec)

        # in principle, TF does not complain about this
        # but we still decided against backprop though matching
        # procedure itself because we do not really understand
        # what will happen
        positive_vec = tf.stop_gradient(positive_vec)
        gt_bboxes_vec = tf.stop_gradient(gt_bboxes_vec)
        cats_vec = tf.stop_gradient(cats_vec)

        # compute all refinements
        refine_vec = encode_bboxes_tf(self.tiling, gt_bboxes_vec, self.config)

        return (positive_vec, cats_vec, refine_vec)
