import tensorflow as tf
import numpy as np
import logging
import logging.config
import time

from config import get_logging_config, args, evaluation_logfile, train_dir
from config import config as net_config
from paths import CKPT_ROOT
from utils import decode_bboxes, batch_iou

from skimage.transform import resize as imresize

from resnet import ResNet
from boxer import PriorBoxGrid
from voc_loader import VOCLoader

logging.config.dictConfig(get_logging_config(args.run_name))
log = logging.getLogger()


def main(argv=None):  # pylint: disable=unused-argument
    net = ResNet
    depth = 50

    loader = VOCLoader('07', 'test')

    net = net(config=net_config, depth=depth, training=False)

    num_classes = 21
    batch_size = args.batch_size
    img_size = args.image_size
    image_ph = tf.placeholder(shape=[1, img_size, img_size, 3],
                              dtype=tf.float32, name='img_ph')
    net.create_trunk(image_ph)
    bboxer = PriorBoxGrid(net_config)

    net.create_multibox_head(num_classes)
    confidence = tf.nn.softmax(tf.squeeze(net.outputs['confidence']))
    location = tf.squeeze(net.outputs['location'])

    good_bboxes = decode_bboxes(location, bboxer.tiling)
    detection_list = []
    score_list = []
    for i in range(1, num_classes):
        class_mask = tf.greater(confidence[:, i], args.conf_thresh)

        class_scores = tf.boolean_mask(confidence[:, i], class_mask)
        class_bboxes = tf.boolean_mask(good_bboxes, class_mask)

        K = tf.minimum(tf.size(class_scores), args.top_k_nms)
        _, top_k_inds = tf.nn.top_k(class_scores, K)
        top_class_scores = tf.gather(class_scores, top_k_inds)
        top_class_bboxes = tf.gather(class_bboxes, top_k_inds)

        final_inds = tf.image.non_max_suppression(top_class_bboxes,
                                                    top_class_scores,
                                                    max_output_size=50,
                                                    iou_threshold=args.nms_thresh)
        final_class_bboxes = tf.gather(top_class_bboxes, final_inds)
        final_scores = tf.gather(top_class_scores, final_inds)

        detection_list.append(final_class_bboxes)
        score_list.append(final_scores)

    net.create_segmentation_head(num_classes)
    segmentation = tf.cast(tf.argmax(tf.squeeze(net.outputs['segmentation']),
                                            axis=-1), tf.int32)
    times = []

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:


        sess.run(tf.global_variables_initializer())

        ckpt_path = train_dir + '/model.ckpt-%i000' % args.ckpt
        log.debug("Restoring checkpoint %s" % ckpt_path)
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, ckpt_path)
        for i in range(200):
            im = loader.load_image(loader.get_filenames()[i])
            im = imresize(im, (img_size, img_size))
            im = im.reshape((1, img_size, img_size, 3))
            st = time.time()
            sess.run([detection_list, score_list, segmentation], feed_dict={image_ph: im})
            et = time.time()
            if i > 10:
                times.append(et-st)
    m = np.mean(times)
    s = np.std(times)
    fps = 1/m
    log.info("Mean={0:.2f}ms; Std={1:.2f}ms; FPS={2:.1f}".format(m*1000, s*1000, fps))



if __name__ == '__main__':
    tf.app.run()
