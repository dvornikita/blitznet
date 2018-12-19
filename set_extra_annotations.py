import os
import sys
import subprocess
import progressbar

import numpy as np
import scipy.io as sio

from shutil import copyfile
from glob import glob
from PIL import Image

from paths import DATASETS_ROOT, EVAL_DIR

def makedir(name):
    if not os.path.exists(name):
        os.makedirs(name)

# List of files that have extra annotations is placed in the dataset folder
print(' - Locating the files')
extra_annot_dir = os.path.join(DATASETS_ROOT, 'VOCdevkit/VOC2012/ImageSets/SegmentationAug/')
makedir(extra_annot_dir)
copyfile(os.path.join(EVAL_DIR, 'Extra', 'train_extra_annot.txt'),
         os.path.join(extra_annot_dir, 'train.txt'))

# Downloading extra data and extracting it
print(' - Downloading extra data')
data_link = 'https://drive.google.com/uc?export=download&id=1EQSKo5n2obj7tW8RytYTJ-eEYbXqtUXE'
archive_name = os.path.join(DATASETS_ROOT, 'benchmark.tgz')
extra_folder_name = os.path.join(DATASETS_ROOT, 'benchmark')
if not os.path.exists(archive_name):
    subprocess.call('wget -P %s %s' % (DATASETS_ROOT, data_link), shell=True)
makedir(extra_folder_name)
if not os.path.exists(extra_folder_name):
    print(' - Unpacking, it may take a while')
    subprocess.call('tar -xf %s -C %s' % (archive_name, extra_folder_name), shell=True)

# Extracting extra annotations to the dataset folder
print(' - Converting data to .png and saving to the dataset folder')
extra_annot_folder = os.path.join(DATASETS_ROOT, 'VOCdevkit/VOC2012/SegmentationClassAug/')
folder_name = os.path.join(extra_folder_name, 'benchmark_RELEASE/dataset/cls')
filenames = glob(os.path.join(folder_name, '*.mat'))
makedir(extra_annot_folder)

palette = np.load(os.path.join(EVAL_DIR, 'Extra/palette.npy')).tolist()
bar = progressbar.ProgressBar()
for i in bar(range(len(filenames))):
    filename = filenames[i]
    name = filename.split('/')[-1].split('.')[0]
    mat = sio.loadmat(filename)['GTcls'][0][0][1]
    mask = Image.fromarray(mat)
    mask.putpalette(palette)
    mask.save(os.path.join(extra_annot_folder, name + '.png'), 'PNG')


# Deleting useless files
print(' - Deleting useless files')
subprocess.call('rm %s' % archive_name)
subprocess.call('rm -r %s' % extra_folder_name)
