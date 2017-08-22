import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from collections import defaultdict, namedtuple
from voc_loader import VOCLoader


Detection = namedtuple('Detection', ['cat', 'score', 'bbox'])


def read_cache_detections(loader):
    path_cache = '/home/lear/kshmelko/gpu_scratch/datasets/voc07/results/cache/%s.pkl' % args.net

    try:
        with open(path_cache, 'rb') as f:
            return pickle.load(f)
    except:
        if args.net == 'frcnn':
            path = '/home/lear/kshmelko/scratch/datasets/voc/VOCdevkit/results/VOC2007/Main'
            prefix = '7a0f6af1-d09a-4931-b1ce-1641ef8c3429_'
        else:
            prefix = ''
        if args.net == 'ssd300':
            path = '/home/lear/kshmelko/scratch/datasets/voc/VOCdevkit/results/VOC2007/SSD_300x300_score/Main'
        if args.net == 'ssd512':
            path = '/home/lear/kshmelko/scratch/datasets/voc/VOCdevkit/results/VOC2007/SSD_512x512_score/Main'
        if args.net == 'blitz300':
            path = '/home/lear/kshmelko/scratch/datasets/voc/VOCdevkit/results/resskip300detseg_voc0712segfull_nonshared_x4_base64_filter3_45-60-75_b32/results/VOC2007/Main'
        if args.net == 'blitz512':
            path = '/home/lear/kshmelko/scratch/datasets/voc/VOCdevkit/results/resskip512detseg_voc0712segfull_nonshared_base64_filter1_35-50-65_b16/results/VOC2007/Main'
        if args.net == 'blitz300-rpn':
            path = '/home/lear/kshmelko/scratch/datasets/voc/VOCdevkit/results/resskip300det_voc0712_nonshared_rpn_b32/results/VOC2007/Main'

        results = defaultdict(list)
        fun = lambda x: int(round(float(x)))

        for cat in loader.categories:
            with open(path+'/comp4_%sdet_test_%s.txt' % (prefix, cat), 'r') as f:
                for line in f.read().split('\n'):
                    if line == '':
                        continue
                    img, score, x, y, x2, y2 = line.split(' ')
                    x, y, x2, y2 = tuple(map(fun, [x, y, x2, y2]))
                    w = x2-x
                    h = y2-y
                    score = float(score)
                    results[img].append(Detection(cat=loader.cats_to_ids[cat],
                                                score=score, bbox=(x, y, w, h)))

        with open(path_cache, 'wb') as f:
            pickle.dump(results, f)

        return results


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

            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
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


def eval_category(cid, gt, dets):
    cgt = gt[cid]
    cdets = np.array(dets[cid])
    if (cdets.shape == (0, )):
        return None
    scores = cdets[:, 1]
    sorted_inds = np.argsort(-scores)
    image_ids = cdets[sorted_inds, 0].astype(int)
    BB = cdets[sorted_inds]

    npos = 0
    for img_gt in cgt.values():
        img_gt['ignored'] = np.array(img_gt['difficult'])
        img_gt['det'] = np.zeros(len(img_gt['difficult']), dtype=np.bool)
        img_gt['score'] = np.zeros(len(img_gt['difficult']), dtype=np.float32)
        npos += np.sum(~img_gt['ignored'])

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    # TODO record matching info
    # print('nd=%i' % nd)
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

        if ovmax > 0.5:
            if not R['ignored'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = True
                    R['score'][jmax] = BB[d, 1]
                    # R['match'][jmax] = bb
                else:
                    fp[d] = 1.
            else:
                R['det'][jmax] = True
                R['score'][jmax] = BB[d, 1]
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
    print('----')
    print(rec)
    print(prec)
    # return rec, prec
    ap = voc_ap(rec, prec, True)
    print(ap)
    print('----')
    return ap


def draw(img, dets, cats, scores, dets_eval, gt_eval, img_id):
    gt_match = []
    gt_cats = []
    gt_bboxes = []
    gt_diff = []
    gt_match_score = []
    for cat in gt_eval:
        gt = gt_eval[cat]
        if len(gt) == 0:
            continue
        gt = gt[img_id]
        for j in range(len(gt['bbox'])):
            gt_cats.append(cat)
            gt_bboxes.append(gt['bbox'][j])
            gt_match.append(gt['det'][j])
            gt_diff.append(gt['difficult'][j])
            gt_match_score.append(gt['score'][j])

    iou_mask = batch_iou(dets, gt_bboxes) >= 0.5
    score_mat = np.zeros_like(iou_mask, dtype=np.float32)
    for i in range(len(dets)):
        score_mat[i, np.where(gt_cats == cats[i])[0]] = scores[i]
    score_mat = score_mat * iou_mask
    matched_det = np.sum(score_mat, axis=1) > 1e-4

    h, w = img.shape[:2]
    image = Image.fromarray((img * 255).astype('uint8'))
    dr = ImageDraw.Draw(image)

    for i in range(len(cats)):
        cat = cats[i]
        score = scores[i]
        if score < args.min_score:
            continue

        bbox = np.array(dets[i])

        bbox[[2, 3]] += bbox[[0, 1]]
        color = 'green' if matched_det[i] else 'red'
        bbox = list(bbox)
        dr.rectangle(bbox, outline=color)
        dr.text(bbox[:2], loader.ids_to_cats[cat] + ' ' + str(score), fill=color)

    for i in range(len(gt_cats)):
        x, y, w, h = gt_bboxes[i]
        color = 'white' if gt_match[i] else 'blue'
        if gt_diff[i]:
            color = 'black'
        dr.rectangle((x, y, x + w, y + h), outline=color)
        dr.text((x, y), loader.ids_to_cats[gt_cats[i]] + ' ' + str(gt_match_score[i]), fill=color)
    plt.title("Network %s, image %s" % (args.net, img_id))
    plt.imshow(np.array(image))
    if not args.noshow:
        plt.show()
    if args.dump_folder != '':
        plt.axis('off')
        plt.savefig(args.dump_folder+'/%06d.jpg' % img_id, bbox_inches='tight')
    del dr


parser = argparse.ArgumentParser(description='Analyze detection results of various networks')
parser.add_argument("--min_score", default=0.6, type=float)
parser.add_argument("--max_map", default=1.5, type=float)
parser.add_argument("--net", required=True, choices=['ssd300', 'ssd512', 'frcnn', 'blitz300', 'blitz512', 'blitz300-rpn'])
# parser.add_argument("--x4", default=False, action='store_true')
parser.add_argument("--images", default='', type=str)
parser.add_argument("--write_difficult", default=False, action='store_true')
parser.add_argument("--noshow", default=False, action='store_true')
parser.add_argument("--dump_folder", default='', type=str)
args = parser.parse_args()

show_img = not args.noshow or args.dump_folder != ''

if __name__ == '__main__':
    loader = VOCLoader('07', 'test')
    results = read_cache_detections(loader)
    maps = []
    difficult_ids = []
    if args.images == '':
        images = loader.get_filenames()
    else:
        with open(args.images, 'r') as f:
            images = f.read().split('\n')
        results = {k: results[k] for k in results if k in images}
    for i in range(len(results)):
        img_id = images[i]
        print(img_id)
        gt_bboxes, _, gt_cats, w, h, difficulty = loader.read_annotations(img_id)
        # print('==============================')
        # print("GT: ", ' '.join(loader.ids_to_cats[j] for j in np.unique(gt_cats)))
        img = loader.load_image(img_id)

        gt = {cid: {} for cid in range(1, loader.num_classes)}
        dets = {cid: [] for cid in range(1, loader.num_classes)}
        for cid in np.unique(gt_cats):
            mask = (gt_cats == cid)
            bbox = gt_bboxes[mask]
            diff = difficulty[mask]
            det = np.zeros(len(diff), dtype=np.bool)
            mscore = np.zeros(len(diff), dtype=np.float32)
            gt[cid][int(img_id)] = {'bbox': bbox, 'difficult': diff, 'det': det, 'score': mscore}

        for d in results[img_id]:
            dets[d.cat].append((int(img_id), d.score, ) + d.bbox)
            # if d.score > 0.6:
            #     print(loader.ids_to_cats[d.cat], d.score, d.bbox)

        aps = []
        # print('=======================')
        # for cat in np.unique([d.cat for d in results[img_id] if d.cat != 0]):
        for cat in np.unique(gt_cats):
            ap = eval_category(cat, gt, dets)
            if ap is not None and not np.all(gt[cat][int(img_id)]['difficult']):
                print("%s\t%.3f" % (loader.ids_to_cats[cat], ap))
                aps.append(ap)
            mAP = np.mean(aps)
            maps.append(mAP)
        if mAP <= args.max_map:
            res = results[img_id]
            print("image %s mAP = %f" % (img_id, mAP))
            dets_im = np.array([d.bbox for d in res]).reshape([-1, 4])
            cats_im = np.array([d.cat for d in res])
            scores_im = np.array([d.score for d in res])
            if args.write_difficult:
                difficult_ids.append(img_id)
            if show_img:
                draw(img, dets_im, cats_im, scores_im, dets, gt, int(img_id))

    if args.write_difficult:
        with open('difficult_%s' % args.net, 'w') as f:
            f.write('\n'.join(difficult_ids))

    # maps = np.array(maps)
    # maps = maps[~np.isnan(maps)]
    # maps = maps[maps < 0.95]
    # print(np.mean(maps), np.min(maps), sorted(maps)[:10])
    # plt.hist(maps, bins=50)
    # plt.show()
