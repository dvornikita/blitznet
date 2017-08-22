import numpy as np
import progressbar
import config
import logging
import json
import os

from tabulate import tabulate
from config import args
from pycocotools.cocoeval import COCOeval

log = logging.getLogger()


class Evaluation(object):
    def __init__(self, detector, loader, iou_thresh=0.5):
        self.detector = detector
        self.loader = loader
        self.gt = {}
        self.dets = {}
        self.iou_thresh = iou_thresh

    def evaluate_network(self, ckpt):
        filenames = self.loader.get_filenames()

        self.gt = {cid: {} for cid in range(1, self.loader.num_classes)}
        self.dets = {cid: [] for cid in range(1, self.loader.num_classes)}

        bar = progressbar.ProgressBar()
        for i in bar(range(len(filenames))):
            self.process_image(filenames[i], i)

        table = self.make_detection_table() if args.detect else None
        iou = self.compute_mean_iou() if args.segment else None

        return self.compact_results(table, ckpt, iou)

    def compute_ap(self):
        """computes average precision for all categories"""
        aps = {}
        for cid in range(1, self.loader.num_classes):
            cat_name = self.loader.ids_to_cats[cid]
            rec, prec = self.eval_category(cid)
            ap = voc_ap(rec, prec, self.loader.year == '07')
            aps[self.loader.ids_to_cats[cid]] = ap
        return aps

    def make_detection_table(self):
        """creates a table with AP per category and mean AP"""
        aps = self.compute_ap()
        eval_cache = [aps]

        table = []
        for cid in range(1, self.loader.num_classes):
            cat_name = self.loader.ids_to_cats[cid]
            table.append((cat_name, ) + tuple(aps.get(cat_name, 'N/A') for aps in eval_cache))
        mean_ap = np.mean([a for a in list(aps.values()) if a >= 0])
        table.append(("AVERAGE", ) + tuple(np.mean(list(aps.values())) for aps in eval_cache))
        x = tabulate(table, headers=(["Category", "mAP (all)"]),
                     tablefmt='orgtbl', floatfmt=".3f")
        log.info("Eval results:\n%s", x)
        return table

    def compact_results(self, table, ckpt, iou=None):
        """compresses the table for concise metrics representation
        during batch evaluation"""
        out = [str(ckpt)]
        if table:
            maps = table[-1][1:]
            out += ['%.3f%%' % maps[0]]
        if iou:
            out += [' %.3f%%' % iou + ' mIoU']
        s = '\t'.join(out) + '\n'
        return s

    def compute_mean_iou(self):
        iou = self.detector.get_mean_iou()
        print(iou)
        log.info("\n Mean IoU is %f", iou)
        return iou

    def process_image(self, name, img_id):
        img = self.loader.load_image(name)
        gt_bboxes, seg_gt, gt_cats, w, h, difficulty = self.loader.read_annotations(name)

        for cid in np.unique(gt_cats):
            mask = (gt_cats == cid)
            bbox = gt_bboxes[mask]
            diff = difficulty[mask]
            det = np.zeros(len(diff), dtype=np.bool)
            self.gt[cid][img_id] = {'bbox': bbox, 'difficult': diff, 'det': det}

        output = self.detector.feed_forward(img, seg_gt, w, h, name,
                                            gt_bboxes, gt_cats,
                                            img_id < args.save_first_n)

        if args.detect:
            det_bboxes, det_probs, det_cats = output[:3]
            for i in range(len(det_cats)):
                self.dets[det_cats[i]].append((img_id, det_probs[i]) + tuple(det_bboxes[i]))

    def eval_category(self, cid):
        """Computes average precision for one category"""
        cgt = self.gt[cid]
        cdets = np.array(self.dets[cid])
        if (cdets.shape == (0, )):
            return None, None
        scores = cdets[:, 1]
        sorted_inds = np.argsort(-scores)
        image_ids = cdets[sorted_inds, 0].astype(int)
        BB = cdets[sorted_inds]

        npos = 0
        for img_gt in cgt.values():
            img_gt['ignored'] = np.array(img_gt['difficult'])
            img_gt['det'] = np.zeros(len(img_gt['difficult']), dtype=np.bool)
            npos += np.sum(~img_gt['ignored'])

        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            ovmax = -np.inf
            if image_ids[d] in cgt:
                R = cgt[image_ids[d]]
                bb = BB[d, 2:].astype(float)

                BBGT = R['bbox'].astype(float)

                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 0] + BBGT[:, 2], bb[0] + bb[2])
                iymax = np.minimum(BBGT[:, 1] + BBGT[:, 3], bb[1] + bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih

                # union
                uni = (bb[2] * bb[3] + BBGT[:, 2] * BBGT[:, 3] - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > self.iou_thresh:
                if not R['ignored'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = True
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        N = float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = rec * N / np.maximum(rec * N + fp, np.finfo(np.float32).eps)
        return rec, prec

    def reset(self):
        self.gt = {}
        self.dets = {}


class COCOEval(Evaluation):
    def __init__(self, net, loader):
        super().__init__(net, loader)

    def process_image(self, img_id, number):
        img = self.loader.load_image(img_id)
        gt_bboxes, gt_cats, w, h, _ = self.loader.read_annotations(img_id)
        seg_gt = self.loader.get_semantic_segmentation(img_id) if args.segment else None
        out = self.detector.feed_forward(img, seg_gt, w, h, img_id,
                                         gt_bboxes, gt_cats, False)
        detections = []
        if args.detect:
            det_bboxes, det_probs, det_cats = out[:3]
            for j in range(len(det_cats)):
                obj = {}
                obj['bbox'] = list(map(float, det_bboxes[j]))
                obj['score'] = float(det_probs[j])
                obj['image_id'] = img_id
                obj['category_id'] = self.loader.ids_to_coco_ids[det_cats[j]]
                detections.append(obj)
        return detections

    def compute_ap(self):
        coco_res = self.loader.coco.loadRes(self.filename)

        cocoEval = COCOeval(self.loader.coco, coco_res)
        cocoEval.params.imgIds = self.loader.get_filenames()
        cocoEval.params.useSegm = False

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        return cocoEval

    def compact_results(self, stats, ckpt):
        out = [str(ckpt)]
        metrics = ['@0.5:0.95', '@0.5', '@0.75', 'small', 'medium', 'large']
        out += ['%.4f%%(%s)' % (val, desc) for val, desc in zip(stats, metrics)]
        s = '\t'.join(out) + '\n'
        return s

    def evaluate_network(self, ckpt):
        path = config.EVAL_DIR + '/Data/'
        self.filename = path + 'coco_%s_%s_%i.json' % (self.loader.split, args.run_name, ckpt)
        detections = []
        filenames = self.loader.get_filenames()

        bar = progressbar.ProgressBar()
        for i in bar(range(len(filenames))):
            img_id = filenames[i]
            detections.extend(self.process_image(img_id, i))
        with open(self.filename, 'w') as f:
            json.dump(detections, f)
        if args.segment:
            iou = self.compute_mean_iou()
        cocoEval = self.compute_ap()
        return self.compact_results(cocoEval.stats, ckpt)


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            p = 0 if np.sum(rec >= t) == 0 else np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
