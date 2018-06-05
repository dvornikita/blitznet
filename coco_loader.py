from paths import DATASETS_ROOT

from pycocotools.coco import COCO
from pycocotools import mask

import cv2
import numpy as np

import logging

log = logging.getLogger()

COCO_VOC_CATS = ['__background__', 'airplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table',
                 'dog', 'horse', 'motorcycle', 'person', 'potted plant',
                 'sheep', 'couch', 'train', 'tv']

COCO_NONVOC_CATS = ['apple', 'backpack', 'banana', 'baseball bat',
                    'baseball glove', 'bear', 'bed', 'bench', 'book', 'bowl',
                    'broccoli', 'cake', 'carrot', 'cell phone', 'clock', 'cup',
                    'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee',
                    'giraffe', 'hair drier', 'handbag', 'hot dog', 'keyboard',
                    'kite', 'knife', 'laptop', 'microwave', 'mouse', 'orange',
                    'oven', 'parking meter', 'pizza', 'refrigerator', 'remote',
                    'sandwich', 'scissors', 'sink', 'skateboard', 'skis',
                    'snowboard', 'spoon', 'sports ball', 'stop sign',
                    'suitcase', 'surfboard', 'teddy bear', 'tennis racket',
                    'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light',
                    'truck', 'umbrella', 'vase', 'wine glass', 'zebra']

COCO_CATS = COCO_VOC_CATS+COCO_NONVOC_CATS

coco_ids = {'airplane': 5, 'apple': 53, 'backpack': 27, 'banana': 52,
            'baseball bat': 39, 'baseball glove': 40, 'bear': 23, 'bed': 65,
            'bench': 15, 'bicycle': 2, 'bird': 16, 'boat': 9, 'book': 84,
            'bottle': 44, 'bowl': 51, 'broccoli': 56, 'bus': 6, 'cake': 61,
            'car': 3, 'carrot': 57, 'cat': 17, 'cell phone': 77, 'chair': 62,
            'clock': 85, 'couch': 63, 'cow': 21, 'cup': 47, 'dining table':
            67, 'dog': 18, 'donut': 60, 'elephant': 22, 'fire hydrant': 11,
            'fork': 48, 'frisbee': 34, 'giraffe': 25, 'hair drier': 89,
            'handbag': 31, 'horse': 19, 'hot dog': 58, 'keyboard': 76, 'kite':
            38, 'knife': 49, 'laptop': 73, 'microwave': 78, 'motorcycle': 4,
            'mouse': 74, 'orange': 55, 'oven': 79, 'parking meter': 14,
            'person': 1, 'pizza': 59, 'potted plant': 64, 'refrigerator': 82,
            'remote': 75, 'sandwich': 54, 'scissors': 87, 'sheep': 20, 'sink':
            81, 'skateboard': 41, 'skis': 35, 'snowboard': 36, 'spoon': 50,
            'sports ball': 37, 'stop sign': 13, 'suitcase': 33, 'surfboard':
            42, 'teddy bear': 88, 'tennis racket': 43, 'tie': 32, 'toaster':
            80, 'toilet': 70, 'toothbrush': 90, 'traffic light': 10, 'train':
            7, 'truck': 8, 'tv': 72, 'umbrella': 28, 'vase': 86, 'wine glass':
            46, 'zebra': 24}
coco_ids_to_cats = dict(map(reversed, list(coco_ids.items())))


class COCOLoader():
    cats_to_ids = dict(map(reversed, enumerate(COCO_CATS)))
    ids_to_cats = dict(enumerate(COCO_CATS))
    num_classes = len(COCO_CATS)
    categories = COCO_CATS[1:]

    def __init__(self, split, memoization=False):
        assert not memoization
        self.dataset = 'coco'
        self.coco_ids_to_internal = {k: self.cats_to_ids[v] for k, v in coco_ids_to_cats.items()}
        self.ids_to_coco_ids = dict(map(reversed, self.coco_ids_to_internal.items()))
        self.split = split
        assert self.split in ['train2014', 'val2014', 'test2014', 'test2015', 'minival2014', 'valminusminival2014', 'test-dev2015']
        self.root = os.path.join(DATASETS_ROOT, 'coco')
        self.included_coco_ids = list(coco_ids.values())
        if 'test' in self.split:
            json = '%s/annotations/image_info_%s.json'
        else:
            json = '%s/annotations/instances_%s.json'
        self.coco = COCO(json % (self.root, self.split))
        self.filenames = self.coco.getImgIds()
        self.num_classes = COCOLoader.num_classes
        self.real_split = self.split
        if self.real_split in ['minival2014', 'valminusminival2014']:
            self.real_split = 'val2014'
            self.coco = COCO('%s/annotations/instances_%s.json' % (self.root, self.real_split))
        if 'test' in self.real_split:
            self.real_split = 'test2015'
        log.info("Created a COCO loader %s with %i images" % (split, len(self.coco.getImgIds())))

    def load_image(self, img_id):
        img = self.coco.loadImgs(img_id)[0]
        img_str = '%s/images/%s/%s' % (self.root, self.real_split, img['file_name'])
        if self.split == 'test-dev2015':
            img_str = img_str.replace('test-dev2015', 'test2015')
        im = cv2.imread(img_str, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)/255.0
        im = im.astype(np.float32)
        return im

    def get_filenames(self):
        # strictly speaking those are not filenames,
        # but the usage is consistent in this class
        return self.filenames

    def _get_coco_annotations(self, img_id, only_instances=True):
        iscrowd = False if only_instances else None
        return self.coco.loadAnns(self.coco.getAnnIds(
            imgIds=img_id, catIds=self.included_coco_ids, iscrowd=iscrowd))

    def read_annotations(self, img_id):
        anns = self._get_coco_annotations(img_id)

        bboxes = [ann['bbox'] for ann in anns]
        cats = [coco_ids_to_cats[ann['category_id']] for ann in anns]
        labels = [COCOLoader.cats_to_ids[cat_name] for cat_name in cats]

        img = self.coco.loadImgs(img_id)[0]

        # removed rounding to int32
        return np.array(bboxes).reshape((-1, 4)), np.array(labels),\
            img['width'], img['height'], np.zeros_like(labels, dtype=np.bool)

    def _read_segmentation(self, ann, H, W):
        s = ann['segmentation']
        s = s if type(s) == list else [s]
        return mask.decode(mask.frPyObjects(s, H, W)).max(axis=2)

    def get_semantic_segmentation(self, img_id):
        img = self.coco.loadImgs(img_id)[0]
        h, w = img['height'], img['width']
        segmentation = np.zeros((h, w), dtype=np.uint8)
        coco_anns = self._get_coco_annotations(img_id, only_instances=False)
        for ann in coco_anns:
            mask = self._read_segmentation(ann, h, w)
            cid = self.coco_ids_to_internal[ann['category_id']]
            assert mask.shape == segmentation.shape
            segmentation[mask > 0] = cid
        return segmentation
