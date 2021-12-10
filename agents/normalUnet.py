import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvutils

from datasets import *
from models.networks import Unet, Discriminator

from utils.im_util import denorm, write_labels_on_images
import numpy as np
from agents.trainingModule import TrainingModule



class NormalUnet(TrainingModule):
    def __init__(self, config):
        super(NormalUnet, self).__init__(config)

        self.G = Unet(conv_dim=config.g_conv_dim,n_layers=config.g_layers,max_dim=config.max_conv_dim, im_channels=config.img_channels, skip_connections=config.skip_connections, vgg_like=config.vgg_like)
        self.D = Discriminator(image_size=config.image_size, im_channels=3, attr_dim=1, conv_dim=config.d_conv_dim,n_layers=config.d_layers,max_dim=config.max_conv_dim,fc_dim=config.d_fc_dim)

        print(self.G)
        if self.config.use_image_disc:
            print(self.D)


        self.data_loader = globals()['{}_loader'.format(self.config.dataset)](
            self.config.data_root, self.config.mode, self.config.attrs,
            self.config.crop_size, self.config.image_size, self.config.batch_size, self.config.data_augmentation)

        self.logger.info("NormalUnet ready")



    ################################################################
    ###################### SAVE/lOAD ###############################
    def save_checkpoint(self):
        self.save_one_model(self.G,self.optimizer_G,'G')
        if self.config.use_image_disc:
            self.save_one_model(self.D,self.optimizer_D,'D')

    def load_checkpoint(self):
        if self.config.checkpoint is None:
            return

        self.load_one_model(self.G,self.optimizer_G if self.config.mode=='train' else None,'G')
        if (self.config.use_image_disc):
            self.load_one_model(self.D,self.optimizer_D if self.config.mode=='train' else None,'D')

        self.current_iteration = self.config.checkpoint


    ################################################################
    ################### OPTIM UTILITIES ############################

    def setup_all_optimizers(self):
        self.optimizer_G = self.build_optimizer(self.G, self.config.g_lr)
        self.optimizer_D = self.build_optimizer(self.D, self.config.d_lr)
        self.load_checkpoint() #load checkpoint if needed 
        self.lr_scheduler_G = self.build_scheduler(self.optimizer_G)
        self.lr_scheduler_D = self.build_scheduler(self.optimizer_D,not(self.config.use_image_disc))

    def step_schedulers(self,scalars):
        self.lr_scheduler_G.step()
        self.lr_scheduler_D.step()
        scalars['lr/g_lr'] = self.lr_scheduler_G.get_lr()[0]
        scalars['lr/d_lr'] = self.lr_scheduler_D.get_lr()[0]

    def eval_mode(self):
        self.G.eval()
        self.D.eval()
    def training_mode(self):
        self.G.train()
        self.D.train()



    ################################################################
    ##################### EVAL UTILITIES ###########################
    def log_img_reconstruction(self,img,normals,path=None,writer=False):
        img=img.to(self.device)
        normals=normals.to(self.device)
        normals_hat = self.G(img)*normals[:,3:]

        x_concat = torch.cat((img[:,:3],normals[:,:3],normals_hat), dim=-1)

        image = tvutils.make_grid(denorm(x_concat), nrow=1)
        if writer:
            self.writer.add_image('sample', image,self.current_iteration)
        if path:
            tvutils.save_image(image,path)





    ########################################################################################
    #####################                 TRAINING               ###########################
    def training_step(self, batch):
        # ================================================================================= #
        #                            1. Preprocess input data                               #
        # ================================================================================= #
        Ia, normals, _, _ = batch

        Ia = Ia.to(self.device)         # input images
        Ia_3ch = Ia[:,:3]
        normals = normals.to(self.device)

        scalars = {}
        # ================================================================================= #
        #                           2. Train the discriminator                              #
        # ================================================================================= #
        if self.config.use_image_disc:
            self.G.eval()
            self.D.train()

            for _ in range(self.config.n_critic):
                # input is the real normal map
                out_disc_real = self.D(normals[:,:3])
                # fake image normals_hat
                normals_hat = self.G(Ia)
                out_disc_fake = self.D(normals_hat.detach())
                #adversarial losses
                d_loss_adv_real = - torch.mean(out_disc_real)
                d_loss_adv_fake = torch.mean(out_disc_fake)
                # compute loss for gradient penalty
                alpha = torch.rand(Ia.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * normals[:,:3].data + (1 - alpha) * normals_hat.data).requires_grad_(True)
                out_disc = self.D(x_hat)
                d_loss_adv_gp = self.config.lambda_gp * self.gradient_penalty(out_disc, x_hat)
                #full GAN loss
                d_loss_adv = d_loss_adv_real + d_loss_adv_fake + d_loss_adv_gp
                d_loss = self.config.lambda_adv * d_loss_adv
                scalars['D/loss_adv'] = d_loss.item()
                scalars['D/loss_real'] = d_loss_adv_real.item()
                scalars['D/loss_fake'] = d_loss_adv_fake.item()
                scalars['D/loss_gp'] = d_loss_adv_gp.item()

                # backward and optimize
                self.optimize(self.optimizer_D,d_loss)
                # summarize
                scalars['D/loss'] = d_loss.item()



        # ================================================================================= #
        #                              3. Train the generator                               #
        # ================================================================================= #
        self.G.train()
        self.D.eval()

        normals_hat = self.G(Ia)
        g_loss_rec = self.config.lambda_G_rec * self.angular_reconstruction_loss(normals[:,:],normals_hat)
        g_loss = g_loss_rec
        scalars['G/loss_rec'] = g_loss_rec.item()


        if self.config.use_image_disc:
            # original-to-target domain : normals_hat -> GAN + classif
            normals_hat = self.G(Ia)
            out_disc = self.D(normals_hat)
            # GAN loss
            g_loss_adv = - self.config.lambda_adv * torch.mean(out_disc)
            g_loss += g_loss_adv
            scalars['G/loss_adv'] = g_loss_adv.item()

        # backward and optimize
        self.optimize(self.optimizer_G,g_loss)
        # summarize
        scalars['G/loss'] = g_loss.item()

        return scalars

    def validating_step(self, batch):
        Ia, normals, _, _ = batch
        self.log_img_reconstruction(Ia,normals,os.path.join(self.config.sample_dir, 'sample_{}.png'.format(self.current_iteration)),writer=True)


    def testing_step(self, batch, batch_id):
        i, (Ia, normals, _, _) = batch_id, batch
        self.log_img_reconstruction(Ia,normals,os.path.join(self.config.result_dir, 'sample_{}_{}.png'.format(i + 1,self.config.checkpoint)),writer=False)


