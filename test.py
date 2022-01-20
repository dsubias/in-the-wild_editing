import torchvision.utils as tvutils
import argparse
import os
import re

# from agents.faderNet import FaderNet
from utils.config import *
from agents import *
from datasets.material import *
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import pytorch_lightning as pl
from utils.im_util import denorm

# Set resource usage
torch.set_num_threads(8)
os.nice(10)


def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--config',
        default='configs/train_fadernet_withnormals_org.yaml',
        help='The path of configuration file in yaml format')
    args = arg_parser.parse_args()
    config = process_config(args.config)
    model = FaderNetPL(config)

    data_loader = DataModule(config.data_root,
                             config.train_file,
                             config.test_file,
                             config.mode,
                             config.attrs,
                             config.crop_size,
                             config.image_size,
                             config.batch_size,
                             config.data_augmentation,
                             mask_input_bg=config.mask_input_bg)

    test_loader = data_loader.val_dataloader()
    path = 'lightning_logs/generator_1_finish/checkpoints/epoch=399-step=105199.ckpt'
    checkpoint = torch.load(path, map_location=model.device)

    model.load_state_dict(checkpoint['state_dict'])

    max_val = 5.0
    i = 0
    for batch in iter(test_loader):
        path = 'test_{}.png'.format(i)
        i += 1
        model.compute_sample_grid(batch, max_val, path, 0)


if __name__ == '__main__':
    main()
