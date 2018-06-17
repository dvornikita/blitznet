import sys
import numpy as np
from os.path import join as opj
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, '/home/nik/Git/blitznet/')

from paths import EVAL_DIR

VOC_CATS = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
            'tvmonitor']

colors = np.load(opj(EVAL_DIR, 'Extra/colors.npy')).tolist()
font = ImageFont.truetype(opj(EVAL_DIR, "Extra/FreeSansBold.ttf"), 20)


size = 500
step = (size - 100) // 10
canvas = Image.new('RGB', (size, size), (255, 255, 255))
dr = ImageDraw.Draw(canvas)

for i in range(1, 21):
    corner = (100 + size / 2 * (i > 10), 50 * ((i - 1) % 10) + 50)
    category = VOC_CATS[i]
    print(category, corner)
    color = colors[i-1]
    dr.text(corner, category, fill=color, font=font)

canvas.show()
