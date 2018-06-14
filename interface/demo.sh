#!/bin/bash

python3 ../interface/main.py --run_name=BlitzNet512_COCO+VOC07+12 --image_size=512 --ckpt=1 --eval_min_conf=0.5 --detect --segment
