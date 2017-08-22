import tensorflow as tf
import numpy as np
import logging
from config import config
from math import ceil, floor

log = logging.getLogger()


def filter_proposals(proposals, gt_bboxes):
    if len(gt_bboxes) > 0:
        idx = np.where(batch_iou(proposals, gt_bboxes).max(axis=1) < 0.5)[0]
        proposals = proposals[idx]
    return proposals, idx


def zoom_out(img, gt_bboxes, params):
    X_out = np.random.uniform(1.0, params['X_out'])
    h, w, _ = img.shape
    if X_out - 1.0 < 5e-2:
        return img, gt_bboxes

    x_shift = np.random.randint(0, floor((X_out - 1) * w))
    y_shift = np.random.randint(0, floor((X_out - 1) * h))

    img_out = np.zeros([ceil(h * X_out), ceil(w * X_out), 3], dtype='float32')
    img_out[:, :] = params['zoomout_color']
    img_out[y_shift:(y_shift + h), x_shift:(x_shift + w)] = img

    gt_bboxes[:, [0, 1]] += x_shift / w, y_shift / h
    gt_bboxes /= X_out
    return img_out, gt_bboxes


def hflip_rois(bboxes, width):
    x = bboxes[:, 0]
    y = bboxes[:, 1]
    w = bboxes[:, 2]
    h = bboxes[:, 3]
    return np.stack([width - x - w, y, w, h], axis=1)


def batch_iou(proposals, gt):
    bboxes = np.transpose(proposals).reshape((4, -1, 1))
    bboxes_x1 = bboxes[0]
    bboxes_x2 = bboxes[0]+bboxes[2]
    bboxes_y1 = bboxes[1]
    bboxes_y2 = bboxes[1]+bboxes[3]

    gt = np.transpose(gt).reshape((4, 1, -1))
    gt_x1 = gt[0]
    gt_x2 = gt[0]+gt[2]
    gt_y1 = gt[1]
    gt_y2 = gt[1]+gt[3]

    widths = np.maximum(0, np.minimum(bboxes_x2, gt_x2) -
                        np.maximum(bboxes_x1, gt_x1))
    heights = np.maximum(0, np.minimum(bboxes_y2, gt_y2) -
                         np.maximum(bboxes_y1, gt_y1))
    intersection = widths*heights
    union = bboxes[2]*bboxes[3] + gt[2]*gt[3] - intersection
    return (intersection / union)


def decode_bboxes(tcoords, anchors):
    var_x, var_y, var_w, var_h = config['prior_variance']
    t_x = tcoords[:, 0]*var_x
    t_y = tcoords[:, 1]*var_y
    t_w = tcoords[:, 2]*var_w
    t_h = tcoords[:, 3]*var_h
    a_w = anchors[:, 2]
    a_h = anchors[:, 3]
    a_x = anchors[:, 0]+a_w/2
    a_y = anchors[:, 1]+a_h/2
    x = t_x*a_w + a_x
    y = t_y*a_h + a_y
    w = tf.exp(t_w)*a_w
    h = tf.exp(t_h)*a_h

    x1 = tf.maximum(0., x - w/2)
    y1 = tf.maximum(0., y - h/2)
    x2 = tf.minimum(1., w + x1)
    y2 = tf.minimum(1., h + y1)
    return tf.stack([y1, x1, y2, x2], axis=1)


def encode_bboxes(proposals, gt):
    prop_x = proposals[:, 0]
    prop_y = proposals[:, 1]
    prop_w = proposals[:, 2]
    prop_h = proposals[:, 3]

    gt_x = gt[:, 0]
    gt_y = gt[:, 1]
    gt_w = gt[:, 2]
    gt_h = gt[:, 3]

    diff_x = (gt_x + 0.5*gt_w - prop_x - 0.5*prop_w)/prop_w
    diff_y = (gt_y + 0.5*gt_h - prop_y - 0.5*prop_h)/prop_h
    if len(gt) > 0 and (np.min(gt_w/prop_w) < 1e-6 or np.min(gt_h/prop_h) < 1e-6):
        print(np.min(gt_w), np.min(gt_h), np.min(gt_w/prop_w), np.max(gt_h/prop_h))
    diff_w = np.log(gt_w/prop_w)
    diff_h = np.log(gt_h/prop_h)

    var_x, var_y, var_w, var_h = config['prior_variance']
    x = np.stack([diff_x/var_x, diff_y/var_y, diff_w/var_w, diff_h/var_h],
                 axis=1)
    return x


def print_variables(name, var_list, level=logging.DEBUG):
    """Handy tool for printing vars"""
    variables = sorted([v.op.name for v in var_list])
    s = "Variables to %s:\n%s" % (name, '\n'.join(variables))
    if level < 0:
        print(s)
    else:
        log.log(level, s)
