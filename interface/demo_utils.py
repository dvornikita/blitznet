import os
import sys
import requests
import numpy as np

from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from os.path import join as opj

sys.path.insert(0, opj(os.path.dirname(os.path.realpath(__file__)), '../'))

from paths import EVAL_DIR

VOC_CATS = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
            'tvmonitor']
DOWNLOAD_ROOT = '~/Downloads/'


def put_transparent_mask(img, mask, palette):
    mask_np = np.array(mask, dtype=np.uint8)
    mask_png = Image.fromarray(mask_np, mode='P')
    mask_png.putpalette(palette)
    mask_rgb = mask_png.convert('RGB')

    mask_np[mask_np > 0] = 130
    mask_np[mask_np == 0] = 255
    mask_l = Image.fromarray(mask_np, mode='L')
    out_image = Image.composite(img, mask_rgb, mask_l)
    return out_image


def image_on_fixed_canvas(image, size=600):
    w, h = image.size
    scale = np.min([size / w, size / h])
    new_w, new_h = (np.array([w, h]) * scale).astype(int)
    image_to_paste = image.resize((new_w, new_h))
    canvas = Image.new('RGB', (size, size), (255, 255, 255))
    offset = ((size - new_w) // 2, (size - new_h) // 2)
    canvas.paste(image_to_paste, offset)
    return canvas


def download_link(link):
    root = DOWNLOAD_ROOT
    r = requests.get(link)
    img = Image.open(BytesIO(r.content)).convert('RGB')
    random_path = opj(root, str(np.random.randint(0, 99999)) + '.jpg')
    img.save(random_path, 'JPEG')
    return random_path


def make_teaser(size, colors):
    step = (size - 100) // 10
    canvas = Image.new('RGB', (size, size), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    font_size = int(size / 60 + 10)
    font = ImageFont.truetype(opj(EVAL_DIR, "Extra/FreeSansBold.ttf"), font_size)

    for i in range(1, 21):
        corner = (100 + size / 2 * (i > 10), step * ((i - 1) % 10) + step)
        category = VOC_CATS[i]
        color = colors[i]
        dr.text(corner, category, fill=color, font=font)

    return canvas


if __name__ == "__main__":
    link = "https://i5.walmartimages.com/asr/cbba00cf-98c3-4d3d-8b23-6b9f1772645f_1.4ce25e2216395370b934f9605651e0ea.jpeg?odnHeight=450&odnWidth=450&odnBg=FFFFFF"
    proc = download_link(link)
    proc.show()
