from distutils.command.config import config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from tqdm import tqdm, trange
from models.stgan import *
from datasets import *
from tqdm import tqdm
import os
from utils.im_util import denorm
import numpy as np
import cv2
from matplotlib import cm
import argparse
from utils.config import *

DIR = 'data_iters'

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
            '--config',
            default='configs/train_stgan.yaml',
            help='The path of configuration file in yaml format')
args = arg_parser.parse_args()
config = process_config(args.config)

data_loader = globals()['{}_loader'.format(config.dataset)](
                    config.data_root, 
                    config.train_file, 
                    config.test_file, 
                    config.mode, 
                    config.attrs,
                    config.crop_size, 
                    config.image_size, 
                    config.batch_size, 
                    config.data_augmentation, 
                    mask_input_bg=config.mask_input_bg,
                    att_neg=config.att_neg,
                    thres_edition=config.thres_edition)

data_iter = iter(data_loader.train_loader)
if not os.path.exists(DIR):
    os.makedirs(DIR)

b = 0
hue_bins = 360
sat_bins= value_bins = 256
plot = False
count_h = np.zeros(hue_bins, dtype=np.float128)
count_s = np.zeros(sat_bins, dtype=np.float128)
count_v = np.zeros(value_bins, dtype=np.float128)
hue_range = np.linspace(0,hue_bins-1,hue_bins)
sat_range = np.linspace(0,1,sat_bins)
value_range = np.linspace(0,255,value_bins)
colors = cm.hsv( hue_range / float(hue_bins-1))
colors_values  = cm.gray( value_range / float(value_bins-1))

for batch in trange(0, data_loader.train_iterations, desc='Epoch {}'.format(0),leave=(0==config.max_epochs-1)):

    x_real, label_org, rgb = next(data_iter)


    de_norm = denorm(x_real, device='cpu', add_bg=False)
    de_norm = de_norm.cpu().numpy()

    if plot:

        for i in range(de_norm.shape[0]):
            
            fig = plt.figure()
            plt.imshow(np.transpose(de_norm[i], (1, 2, 0)))
            plt.title(str(label_org[i].item()))
            plt.savefig(DIR + '/' + str(b) + '-' + str(i)+".png")
            plt.close(fig)

    else:

        masks = de_norm[:,3] > 0
        masks = masks.astype(np.uint8) * 255
        de_norm = de_norm[:,:3].transpose(0,2,3,1)
        de_norm = de_norm  * 255

        for i in range(de_norm.shape[0]):

            img = cv2.cvtColor(de_norm[i], cv2.COLOR_RGB2HSV)
            h, s, v = img[:,:,0], img[:,:,1], img[:,:,2]
            hist_h = cv2.calcHist([h],[0],masks[i],[hue_bins],[0,359])
            hist_h = hist_h.reshape((360,))

            hist_s = cv2.calcHist([s],[0],masks[i],[sat_bins],[0,1])
            hist_s = hist_s.reshape((sat_bins,))

            hist_v = cv2.calcHist([v],[0],masks[i],[value_bins],[0,255])
            hist_v = hist_v.reshape((value_bins,))

            count_h += hist_h
            count_s += hist_s
            count_v += hist_v

    b+=1

fig, ax = plt.subplots(3,1,figsize=(8,12))
ax[0].set_title('HUE')
ax[0].bar(hue_range,count_h , color = colors)
ax[0].set_facecolor("grey")
ax[1].set_title('SATURATION')
ax[1].bar(sat_range,count_s,width=1 / sat_bins)
ax[2].set_title('VALUE')
ax[2].bar(value_range,count_v,color = colors_values)
ax[2].set_facecolor('coral')
plt.savefig('hist_color.png')
