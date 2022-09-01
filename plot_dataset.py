
from torch.utils.data import DataLoader
from datasets import *
import torch
from utils.config import *
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from utils.im_util import denorm
import numpy as np
from tqdm import tqdm, trange
from torchvision.utils import make_grid, save_image
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
for batch in trange(0, 3, desc='Epoch {}'.format(0),leave=(0==config.max_epochs-1)):

    x_real, label_org, rgb = next(data_iter)


    de_norm = denorm(x_real, device='cpu',add_bg=False)
    de_norm = de_norm.cpu()

    for i in range(de_norm.shape[0]):
        
        fig = plt.figure()
        plt.imshow(np.transpose(de_norm[i], (1, 2, 0)))
        plt.title(str(label_org[i].item()))
        plt.savefig(DIR + '/' + str(b) + '-' + str(i)+".png")
        plt.close(fig)
        save_image(rgb,"org.png")
        save_image(de_norm,"augmented.png")
        exit()
    b+=1