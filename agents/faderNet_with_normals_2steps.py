import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvutils

from datasets import *
from models.networks import FaderNetGeneratorWithNormals2Steps,FaderNetGeneratorWithNormals, Discriminator, Latent_Discriminator, Unet,reshape_and_concat#, MyDataParallel
from modules.perceptual_loss import PerceptualLoss, StyleLoss, VGG16FeatureExtractor

from utils.im_util import denorm, write_labels_on_images
import numpy as np
from agents.faderNet import FaderNet
from utils import resize_right, interp_methods



class FaderNetWithNormals2Steps(FaderNet):
    def __init__(self, config):
        super(FaderNetWithNormals2Steps, self).__init__(config)

        ###only change generator
        self.G = FaderNetGeneratorWithNormals2Steps(conv_dim=config.g_conv_dim,n_layers=config.g_layers,max_dim=config.max_conv_dim, im_channels=config.img_channels, skip_connections=config.skip_connections, vgg_like=config.vgg_like, attr_dim=len(config.attrs), n_attr_deconv=config.n_attr_deconv, n_concat_normals=config.n_concat_normals, normalization=self.norm, first_conv=config.first_conv, n_bottlenecks=config.n_bottlenecks, all_feat=config.all_feat, img_size=self.config.image_size, batch_size=self.config.batch_size)
        print(self.G)

        ### load the normal predictor network
        #self.normal_G = Unet(conv_dim=config.g_conv_dim_normals,n_layers=config.g_layers_normals,max_dim=config.max_conv_dim_normals, im_channels=config.img_channels, skip_connections=config.skip_connections_normals, vgg_like=config.vgg_like_normals)
        #self.load_model_from_path(self.normal_G,config.normal_predictor_checkpoint)
        #self.normal_G.eval()

        #load the small FaderNet
        self.G_small = FaderNetGeneratorWithNormals(conv_dim=32,n_layers=6,max_dim=512, im_channels=config.img_channels, skip_connections=0, vgg_like=0, attr_dim=len(config.attrs), n_attr_deconv=1, n_concat_normals=4, normalization=self.norm, first_conv=False, n_bottlenecks=2)
        #print(self.G_small) 
        self.load_model_from_path(self.G_small,config.faderNet_checkpoint)
        self.G_small.eval()

        self.G_small=self.to_multi_GPU(self.G_small)
        #self.G_small = MyDataParallel(self.G_small, device_ids=list(range(self.config.ngpu)))

        self.logger.info("FaderNet with normals in 2 steps ready")




    ################################################################
    ##################### EVAL UTILITIES ###########################
    def decode(self,bneck,encodings,att):
        normals= self.get_normals()
        fn_output, fn_features, _ = self.get_fadernet_output(att)
        return self.G.decode(att,bneck,normals,fn_features,encodings)

    # def encode(self):
    #     #put attribute with input image
    #     im_and_att=reshape_and_concat(self.batch_Ia,self.batch_a_att)
    #     return self.G.encode(im_and_att)
    # def encode(self):
    #     #give the output of fadernet as input
    #     im,fn_feat_dec,fn_feat_enc=self.get_fadernet_output(self.current_att)
    #     mask=self.batch_Ia[:,3:]
    #     im=torch.cat([im,mask],dim=1)
    #     return self.G.encode(im,fn_feat_enc)

    def init_sample_grid(self):
        x_fake_list = [self.get_fadernet_output(self.batch_a_att)[0],self.batch_Ia[:,:3]]
        x_fake_list = [self.batch_Ia[:,:]]
        return x_fake_list


    def get_normals(self):
        return self.batch_normals[:,:3]
        # with torch.no_grad():
        #     normals=self.normal_G(self.batch_Ia)
        #     return normals*self.batch_Ia[:,3:]


    def get_fadernet_output(self,att):
        with torch.no_grad():
            if self.config.image_size == 128:
                encodings,z,_ = self.G_small.encode(self.batch_Ia)
                fn_output, fn_features= self.G_small.decode_with_features(att,z,self.get_normals(),encodings)
                return fn_output, fn_features, encodings
            else: # self.config.image_size == 256:
                #rescale if input image is size 256
                #rescaled_im=nn.functional.interpolate(self.batch_Ia, mode='bilinear', align_corners=True, scale_factor=0.5)
                #rescaled_normals=nn.functional.interpolate(self.get_normals(), mode='bilinear', align_corners=True, scale_factor=0.5)
                rescaled_im=resize_right.resize(self.batch_Ia, scale_factors=0.5)
                rescaled_normals=resize_right.resize(self.get_normals(), scale_factors=0.5)

                encodings,z,_ = self.G_small.encode(rescaled_im)
                fn_output, fn_features = self.G_small.decode_with_features(att,z,rescaled_normals,encodings)
            
                #fn_output=nn.functional.interpolate(fn_output, mode='bilinear', align_corners=True, scale_factor=2)
                #fn_features=[nn.functional.interpolate(map, mode='bilinear', align_corners=True, scale_factor=2) for map in fn_features]
                fn_output=resize_right.resize(fn_output, scale_factors=2)
                #fn_features=[resize_right.resize(map, scale_factors=2) for map in fn_features]
                return fn_output, fn_features, encodings



