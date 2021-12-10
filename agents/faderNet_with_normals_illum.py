import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvutils

from datasets import *
from models.networks import FaderNetGeneratorWithNormalsAndIllum, Discriminator, Latent_Discriminator, Unet
from modules.perceptual_loss import PerceptualLoss, StyleLoss, VGG16FeatureExtractor

from utils.im_util import denorm, write_labels_on_images
import numpy as np
from agents.faderNet import FaderNet



class FaderNetWithNormalsAndIllum(FaderNet):
    def __init__(self, config):
        super(FaderNetWithNormalsAndIllum, self).__init__(config)

        ###only change generator
        self.G = FaderNetGeneratorWithNormalsAndIllum(conv_dim=config.g_conv_dim,n_layers=config.g_layers,max_dim=config.max_conv_dim, im_channels=config.img_channels, skip_connections=config.skip_connections, vgg_like=config.vgg_like, attr_dim=len(config.attrs), n_attr_deconv=config.n_attr_deconv, n_concat_normals=config.n_concat_normals, n_concat_illum=4, normalization=self.norm)
        print(self.G)

        ### load the normal predictor network
        self.normal_G = Unet(conv_dim=config.g_conv_dim_normals,n_layers=config.g_layers_normals,max_dim=config.max_conv_dim_normals, im_channels=config.img_channels, skip_connections=config.skip_connections_normals, vgg_like=config.vgg_like_normals)
        #self.load_model_from_path(self.normal_G,config.normal_predictor_checkpoint)
        self.normal_G.eval()

        #change data loader to load illuminations
        self.data_loader = globals()['{}_loader'.format(self.config.dataset)](
            self.config.data_root, self.config.mode, self.config.attrs,
            self.config.crop_size, self.config.image_size, self.config.batch_size, self.config.data_augmentation, mask_input_bg=config.mask_input_bg, use_illum=True)


        self.logger.info("FaderNet with normals and illum ready")




    ################################################################
    ##################### EVAL UTILITIES ###########################
    def decode(self,bneck,encodings,att):
        normals= self.get_normals()
        return self.G.decode(att,bneck,normals,self.batch_illum,encodings)

    def init_sample_grid(self):
        x_fake_list = [self.get_normals(),self.batch_illum,self.batch_Ia[:,:3]]
        #x_fake_list = [self.get_normals(),torch.cat([illum,illum,illum],dim=1),self.batch_Ia[:,:3]]
        return x_fake_list


    def get_normals(self):
        return self.batch_normals[:,:3]
        normals=self.normal_G(self.batch_Ia)
        return normals*self.batch_Ia[:,3:]








