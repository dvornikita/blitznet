import os, sys
import shutil
import subprocess
import tensorflow as tf
import numpy as np
import tkinter as tk

from skimage.transform import resize as imresize
from glob import glob
from os.path import join as opj
from PIL import ImageTk, Image, ImageDraw, ImageFont
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror

from demo_utils import (put_transparent_mask, image_on_fixed_canvas,
                        download_link, make_teaser)

# adding the root dir
sys.path.insert(0, opj(os.path.dirname(os.path.realpath(__file__)), '../'))

from paths import EVAL_DIR
from demo import Loader
from detector import Detector
from config import args, train_dir
from config import config as net_config
from resnet import ResNet

init_dir = "~/Downloads"     # where you look up images first
draw_rectangle = Detector.draw_rectangle
colors = np.load(opj(EVAL_DIR, 'Extra/colors.npy')).tolist()
palette = np.load(opj(EVAL_DIR, 'Extra/palette.npy')).tolist()
font = ImageFont.truetype(opj(EVAL_DIR, "Extra/FreeSansBold.ttf"), 16)


class Application(tk.Frame):
    def __init__(self, master=None, sess=None):
        super().__init__(master)
        self.root=master
        self.root.resizable(width=1, height=1)
        self.size = 1200
        self.pack()

        self.create_widgets()

        self.view_classes = True
        self.sess = sess
        self.init_detectot()

    def load_file(self):
        fname = askopenfilename(filetypes=(("jpeg files", "*.jpg"),
                                           ("all files","*.*")),
                                initialdir=init_dir)
        if fname:
            try:
                self.filename = fname
                self.last_path = fname
                self.change_image(path=fname)
            except Exception as e:                     # <- naked except is a bad idea
                showerror("Open Source File", "Failed to read file\n'%s'" % fname)
                print(e)
            return

    def create_widgets(self):
        # Run BlitzNet button
        self.run = tk.Button(self, text='Run BlitzNet',
                             command=self.run_blitznet,
                             fg='green', width=self.size // 8)
        self.run.pack(side="top")

        # Brows button
        self.button = tk.Button(self, text="Browse",
                                command=self.load_file,
                                width=self.size // 8)
        self.button.pack()

        # from clipboard button
        self.clip = tk.Button(self, text="From Clipboard",
                              command=self.from_clipboard,
                              width=self.size // 8)
        self.clip.pack()

        # Quit button
        self.switch = tk.Button(self, text='View Classes', fg="red",
                              width=self.size // 8,
                                command=self.image_switch)
        self.switch.pack(side="bottom")

        # Image to be detected
        # path = '/home/nik/Downloads/lock.jpeg'
        # img = ImageTk.PhotoImage(Image.open(path).resize((self.size, self.size)))
        img = make_teaser(self.size, colors)
        img = ImageTk.PhotoImage(img)
        self.panel = tk.Label(self.root, image=img)
        self.panel.image = img
        self.panel.pack(side = "bottom", fill = "both", expand = "yes")


    def image_switch(self):
        if self.view_classes:
            self.view_classes = False
            self.switch.text = "View Classes"
            self.change_image(path=self.last_path)
        else:
            self.view_classes = True
            self.switch.text = "View Image"
            img = make_teaser(self.size, colors)
            self.change_image(img=img)

    def change_image(self, path=None, img=None):
        img = image_on_fixed_canvas(Image.open(path), self.size) if img is None else img
        img = ImageTk.PhotoImage(img)
        self.panel.configure(image=img)
        self.panel.image = img

    def from_clipboard(self):
        clipboard = self.clipboard_get()
        print(clipboard)
        self.filename = download_link(clipboard)
        self.last_path = self.filename
        self.change_image(path=self.filename)

    def init_detectot(self):
        assert args.detect or args.segment, "Either detect or segment should be True"
        assert args.ckpt > 0, "Specify the number of checkpoint"
        net = ResNet(config=net_config, depth=50, training=False)
        self.loader = Loader(opj(EVAL_DIR, 'demodemo'))
        self.detector = Detector(self.sess, net, self.loader, net_config, no_gt=args.no_seg_gt,
                                 folder=opj(self.loader.folder, 'output'))
        self.detector.restore_from_ckpt(args.ckpt)

    def run_blitznet(self):
        name = self.filename.split('/')[-1].split('.')[0]
        image = self.loader.load_image(path=self.filename)
        h, w = image.shape[:2]
        print('Processing {}'.format(name + self.loader.data_format))
        output = self.detector.feed_forward(img=image, name=name, w=w, h=h, draw=False,
                                            seg_gt=None, gt_bboxes=None, gt_cats=None)
        boxes, scores, cats, mask, _ = output
        proc_img = self.draw(self.filename, name, boxes, scores, cats, mask)
        self.change_image(path=opj(EVAL_DIR, 'demodemo', 'output', name
                                   + '_processed' + self.loader.data_format))
        print('Done')

    def draw(self, img_path, name, dets, scores, cats, mask):
        image = Image.open(img_path)
        w, h = image.size

        mask = imresize(mask, (h, w), order=0, preserve_range=True).astype(int)
        image = put_transparent_mask(image, mask, palette)

        dr = ImageDraw.Draw(image)

        for i in range(len(cats)):
            cat = cats[i]
            score = scores[i]
            bbox = np.array(dets[i])

            bbox[[2, 3]] += bbox[[0, 1]]
            color = colors[cat]
            draw_rectangle(dr, bbox, color, width=5)
            dr.text(bbox[:2], self.loader.ids_to_cats[cat] + ' ' + str(score)[:4],
                    fill=color, font=font)

        path_to_save = opj(EVAL_DIR, 'demodemo', 'output',
                           name + '_processed' + self.loader.data_format)
        image.save(path_to_save, 'JPEG')
        self.last_path = path_to_save
        self.view_classes = False
        return image


def main(argv=None):  # pylint: disable=unused-argument
    root = tk.Tk()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        app = Application(master=root, sess=sess)
        app.mainloop()


if __name__ == '__main__':
    tf.app.run()
