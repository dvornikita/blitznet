import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from config import args


def xywh_to_yxyx(xywh):
    x, y, w, h = tf.unstack(xywh, axis=1)
    return tf.stack([y, x, y+h, x+w], axis=1)


def yxyx_to_xywh(yxyx):
    y1, x1, y2, x2 = tf.unstack(yxyx, axis=1)
    return tf.stack([x1, y1, x2-x1, y2-y1], axis=1)


def photometric_distortions(image, color_ordering, params, scope=None):
    with tf.name_scope(scope, 'distort_color', [image]):
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=params['brightness_delta'])
            image = tf.image.random_saturation(image, lower=1-params['saturation_delta'],
                                               upper=1+params['saturation_delta'])
            image = tf.image.random_hue(image, max_delta=params['hue_delta'])
            image = tf.image.random_contrast(image, lower=1-params['contrast_delta'],
                                             upper=1+params['contrast_delta'])
        elif color_ordering == 1:
            image = tf.image.random_saturation(image, lower=1-params['saturation_delta'],
                                               upper=1+params['saturation_delta'])
            image = tf.image.random_brightness(image, max_delta=params['brightness_delta'])
            image = tf.image.random_contrast(image, lower=1-params['contrast_delta'],
                                             upper=1+params['contrast_delta'])
            image = tf.image.random_hue(image, max_delta=params['hue_delta'])
        elif color_ordering == 2:
            image = tf.image.random_contrast(image, lower=1-params['contrast_delta'],
                                             upper=1+params['contrast_delta'])
            image = tf.image.random_hue(image, max_delta=params['hue_delta'])
            image = tf.image.random_brightness(image, max_delta=params['brightness_delta'])
            image = tf.image.random_saturation(image, lower=1-params['saturation_delta'],
                                               upper=1+params['saturation_delta'])
        elif color_ordering == 3:
            image = tf.image.random_hue(image, max_delta=params['hue_delta'])
            image = tf.image.random_saturation(image, lower=1-params['saturation_delta'],
                                               upper=1+params['saturation_delta'])
            image = tf.image.random_contrast(image, lower=1-params['contrast_delta'],
                                             upper=1+params['contrast_delta'])
            image = tf.image.random_brightness(image, max_delta=params['brightness_delta'])
        else:
            raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def mirror_distortions(image, rois, params):
    x, y, w, h = tf.unstack(rois, axis=1)
    flipped_rois = tf.stack([1.0 - x - w, y, w, h], axis=1)
    return tf.cond(tf.random_uniform([], 0, 1.0) < params['flip_prob'],
                   lambda: (tf.image.flip_left_right(image), flipped_rois),
                   lambda: (image, rois))


def zoomout(image, gt_bboxes, params):
    X_out = tf.random_uniform([], 1.05, params['X_out'])
    h, w, _ = tf.unstack(tf.to_float(tf.shape(image)))
    zoomout_color = params['zoomout_color']+[0]

    bg_color = tf.constant(zoomout_color, dtype=tf.float32)
    x_shift = tf.random_uniform([], 0, (X_out - 1) * w)
    y_shift = tf.random_uniform([], 0, (X_out - 1) * h)
    x2_shift = (X_out - 1) * w - x_shift
    y2_shift = (X_out - 1) * h - y_shift
    # somewhat hacky solution to pad with MEAN_COLOR
    # tf.pad does not support custom constant padding unlike numpy
    image -= bg_color
    image = tf.pad(image, tf.to_int32([[y_shift, y2_shift], [x_shift, x2_shift], [0, 0]]))
    image += bg_color

    gt_x, gt_y, gt_w, gt_h = tf.unstack(gt_bboxes, axis=1)
    gt_bboxes = tf.stack([gt_x + x_shift/w,
                          gt_y + y_shift/h,
                          gt_w, gt_h], axis=1)/X_out
    return image, gt_bboxes


def scale_distortions(image, gt_bboxes, gt_cats, params):
    """Samples a random box according to overlapping
    with gt objects criteria and crops it from an image"""
    image, gt_bboxes = tf.cond(tf.random_uniform([], 0, 1.0) < args.zoomout_prob,
                               lambda: zoomout(image, gt_bboxes, params),
                               lambda: (image, gt_bboxes))
    n_channels = image.shape[-1]

    def tf_random_choice(slices, bbox):
        sample = tf.multinomial(tf.log([[10.]*len(slices)]), 1)
        slices = tf.convert_to_tensor(slices)
        bbox = tf.convert_to_tensor(bbox)
        bbox_begin, bbox_size = tf.unstack(slices[tf.cast(sample[0][0],
                                                          tf.int32)])
        distort_bbox = bbox[tf.cast(sample[0][0], tf.int32)]
        return bbox_begin, bbox_size, distort_bbox

    bboxes = tf.expand_dims(xywh_to_yxyx(gt_bboxes), 0)
    samplers = []
    boxes = []
    for iou in params['sample_jaccards']:
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bboxes,
            min_object_covered=iou,
            aspect_ratio_range=[0.5, 2.0],
            area_range=[0.3, 1.0],
            max_attempts=params['crop_max_tries'],
            use_image_if_no_bounding_boxes=True)
        samplers.append(sample_distorted_bounding_box[:2])
        boxes.append(sample_distorted_bounding_box[2][0][0])
    bbox_begin, bbox_size, distort_bbox = tf_random_choice(samplers, boxes)
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    # Nope TF, you are wrong, cropping does not change channels.
    cropped_image.set_shape([None, None, n_channels])
    y1, x1, y2, x2 = tf.unstack(distort_bbox)

    def check(center, mini, maxi):
        return tf.logical_and((center >= mini), (center <= maxi))
    gt_centers = gt_bboxes[:, :2] + gt_bboxes[:, 2:] / 2
    mask = tf.logical_and(check(gt_centers[:, 0], x1, x2),
                          check(gt_centers[:, 1], y1, y2))
    gt_bboxes = tf.boolean_mask(gt_bboxes, mask)
    gt_cats = tf.boolean_mask(gt_cats, mask)
    w = tf.to_float(x2-x1)
    h = tf.to_float(y2-y1)

    gt_x, gt_y, gt_w, gt_h = tf.unstack(gt_bboxes, axis=1)
    gt_x2 = gt_x + gt_w
    gt_y2 = gt_y + gt_h
    gt_x1_clip = tf.clip_by_value(gt_x - x1, 0, w)/w
    gt_x2_clip = tf.clip_by_value(gt_x2 - x1, 0, w)/w
    gt_y1_clip = tf.clip_by_value(gt_y - y1, 0, h)/h
    gt_y2_clip = tf.clip_by_value(gt_y2 - y1, 0, h)/h
    gt_w_clip = gt_x2_clip - gt_x1_clip
    gt_h_clip = gt_y2_clip - gt_y1_clip
    gt_bboxes = tf.stack([gt_x1_clip, gt_y1_clip, gt_w_clip, gt_h_clip],
                         axis=1)

    return cropped_image, gt_bboxes, gt_cats


def filter_small_gt(gt_bboxes, gt_cats, min_size):
    mask = tf.logical_and(gt_bboxes[:, 2] >= min_size,
                          gt_bboxes[:, 3] >= min_size)
    return tf.boolean_mask(gt_bboxes, mask), tf.boolean_mask(gt_cats, mask)


def data_augmentation(img, gt_bboxes, gt_cats, seg, config):
    params = config['train_augmentation']
    img = apply_with_random_selector(
        img,
        lambda x, ordering: photometric_distortions(x, ordering, params),
        num_cases=4)

    if seg is not None:
        img = tf.concat([img, tf.cast(seg, tf.float32)], axis=-1)

    img, gt_bboxes, gt_cats = scale_distortions(img, gt_bboxes, gt_cats,
                                                params)
    img, gt_bboxes = mirror_distortions(img, gt_bboxes, params)
    # XXX reference implementation also randomizes interpolation method
    img_size = config['image_size']
    img_out = tf.image.resize_images(img[..., :3], [img_size, img_size])
    gt_bboxes, gt_cats = filter_small_gt(gt_bboxes, gt_cats, 2/config['image_size'])

    if seg is not None:
        seg_shape = config['fm_sizes'][0]
        seg = tf.expand_dims(tf.expand_dims(img[..., 3], 0), -1)
        seg = tf.squeeze(tf.image.resize_nearest_neighbor(seg, [seg_shape, seg_shape]))
        seg = tf.cast(tf.round(seg), tf.int64)
    return img_out, gt_bboxes, gt_cats, seg


def batch_iou_tf(proposals, gt):
    bboxes = tf.reshape(tf.transpose(proposals), [4, -1, 1])
    bboxes_x1 = bboxes[0]
    bboxes_x2 = bboxes[0]+bboxes[2]
    bboxes_y1 = bboxes[1]
    bboxes_y2 = bboxes[1]+bboxes[3]

    gt = tf.reshape(tf.transpose(gt), [4, 1, -1])
    gt_x1 = gt[0]
    gt_x2 = gt[0]+gt[2]
    gt_y1 = gt[1]
    gt_y2 = gt[1]+gt[3]

    widths = tf.maximum(0.0, tf.minimum(bboxes_x2, gt_x2) -
                        tf.maximum(bboxes_x1, gt_x1))
    heights = tf.maximum(0.0, tf.minimum(bboxes_y2, gt_y2) -
                         tf.maximum(bboxes_y1, gt_y1))
    intersection = widths*heights
    union = bboxes[2]*bboxes[3] + gt[2]*gt[3] - intersection
    return (intersection / union)


def encode_bboxes_tf(proposals, gt, config):
    """Encode bbox coordinates in a format
    used for computing the loss"""
    prop_x = proposals[..., 0]
    prop_y = proposals[..., 1]
    prop_w = proposals[..., 2]
    prop_h = proposals[..., 3]

    gt_x = gt[..., 0]
    gt_y = gt[..., 1]
    gt_w = gt[..., 2]
    gt_h = gt[..., 3]

    diff_x = (gt_x + 0.5*gt_w - prop_x - 0.5*prop_w)/prop_w
    diff_y = (gt_y + 0.5*gt_h - prop_y - 0.5*prop_h)/prop_h
    diff_w = tf.log(gt_w/prop_w)
    diff_h = tf.log(gt_h/prop_h)

    var_x, var_y, var_w, var_h = config['prior_variance']
    x = tf.stack([diff_x/var_x, diff_y/var_y, diff_w/var_w, diff_h/var_h], -1)
    return x


def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].

    Args:
        x: input Tensor.
        func: Python function to apply.
        num_cases: Python int32, number of cases to sample sel from.

    Returns:
        The result of func(x, sel), where func receives the value of the
        selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]

