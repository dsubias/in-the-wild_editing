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
os.nice(5)

def plot_img(img, normals):
    
    img = img.permute(1, 2, 0).detach() / 2 + 0.5
    normals = normals.permute(1, 2, 0).detach() / 2 + 0.5

    grid = make_grid(img)
    plt.imsave('img.png', grid.numpy(), cmap='gray')
    grid = make_grid(normals)
    plt.imsave('nor.png', grid.numpy(), cmap='gray')


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--config',
        default='configs/train_fadernet_withnormals.yaml',
        help='The path of configuration file in yaml format')
    args = arg_parser.parse_args()
    #args.config = 'configs/train_stgan_mat.yaml'
    config = process_config(args.config)
    model = FaderNetPL(config)
    #agent = globals()['{}'.format(config.network)](config)
    data_loader = MaterialDataLoader(config.data_root,
                                     config.mode, 
                                     config.attrs,
                                     config.crop_size, 
                                     config.image_size,
                                     config.batch_size,
                                     config.data_augmentation, 
                                     mask_input_bg=config.mask_input_bg)
    
    
    train_loader = data_loader.train_loader
    val_loader = data_loader.val_loader

    train_batches = data_loader.train_iterations
    val_batches = data_loader.val_iterations
    
    trainer = pl.Trainer(check_val_every_n_epoch = config.sample_step,
                         devices=[1], 
                         accelerator="gpu",
                         limit_train_batches = train_batches,
                         limit_val_batches = val_batches,
                         max_epochs = config.max_epoch,
                         enable_checkpointing=True)
    trainer.fit(model, train_loader, val_loader)


    #test_data_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=4)

    """
    for imgs, normal, attr, illum in train_loader:
        # from matplotlib import pyplot as plt

        # # for i in range(len(imgs)):
        # #     print (infos[i])
        for i in range(len(imgs)):

            plot_img(imgs[i], normal[i])
        
    """
    exit()
    agent.run()


if __name__ == '__main__':
    main()
