import os, sys
import shutil
import subprocess
import tensorflow as tf
import tkinter as tk

from glob import glob
from os.path import join as opj
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror

sys.path.insert(0, '/home/nik/Git/blitznet/')


from paths import EVAL_DIR
from demo import Loader
from detector import Detector
from config import args, train_dir
from config import config as net_config
from resnet import ResNet

class Application(tk.Frame):
    def __init__(self, master=None, sess=None):
        super().__init__(master)
        self.root=master
        self.root.resizable(width=1, height=1)
        self.pack()

        self.create_widgets()

        self.sess = sess
        self.init_detectot()

    def load_file(self):
        fname = askopenfilename(filetypes=(("jpeg files", "*.jpg"),
                                           ("all files","*.*")),
                                initialdir="~/Downloads")
        if fname:
            try:
                self.filename = fname
                self.change_image(fname)
            except Exception as e:                     # <- naked except is a bad idea
                showerror("Open Source File", "Failed to read file\n'%s'" % fname)
                print(e)
            return

    def create_widgets(self):
        self.hi_there = tk.Button(self)
        self.hi_there["text"] = "Run BlitzNet"
        self.hi_there["command"] = self.run_blitznet
        self.hi_there["fg"] = 'green'
        self.hi_there.pack(side="top")

        self.quit = tk.Button(self, text='Quit', fg="red",
                              command=self.root.destroy)
        self.quit.pack(side="bottom")

        self.button = tk.Button(self, text="Browse", command=self.load_file, width=20)
        self.button.pack()

        # path = '/home/nik/Downloads/canada_street_racer.jpg'
        path = '/home/nik/Downloads/lock.jpeg'
        img = ImageTk.PhotoImage(Image.open(path))
        self.panel = tk.Label(self.root, image=img)
        self.panel.image = img
        self.panel.pack(side = "bottom", fill = "both", expand = "yes")

    def change_image(self, path):
        img = ImageTk.PhotoImage(Image.open(path))
        self.panel.configure(image=img)
        self.panel.image = img

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
        self.detector.feed_forward(img=image, name=name, w=w, h=h, draw=True,
                                   seg_gt=None, gt_bboxes=None, gt_cats=None)
        self.change_image(opj(EVAL_DIR, 'demodemo', 'output',
                              name + '_det_50' + self.loader.data_format))
        print('Done')


def main(argv=None):  # pylint: disable=unused-argument
    root = tk.Tk()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False)) as sess:
        app = Application(master=root, sess=sess)
        app.mainloop()


if __name__ == '__main__':
    tf.app.run()
