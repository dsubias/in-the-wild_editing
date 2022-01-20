import argparse
import os

#from agents.faderNet import FaderNet
from utils.config import *
from agents import *
from datasets.material import *
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import pytorch_lightning as pl

# Set resource usage
torch.set_num_threads(8)
os.nice(10)


def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--config',
        default='configs/train_fadernet_new.yaml',
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

    val_loader = data_loader.val_dataloader()
    train_loader = data_loader.train_dataloader()
    test_loader = data_loader.test_dataloader()

    train_batches = data_loader.train_iterations
    val_batches = data_loader.val_iterations

    trainer = pl.Trainer(check_val_every_n_epoch=config.sample_step,
                         devices=1,
                         accelerator="cpu",
                         log_every_n_steps=10,
                         limit_train_batches=20,
                         limit_val_batches=val_batches,
                         max_epochs=config.max_epoch,
                         enable_checkpointing=True)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
