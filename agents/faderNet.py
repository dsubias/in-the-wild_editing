import os
import time
import datetime
from numpy.core.function_base import linspace
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvutils

from datasets import *
from models.networks import FaderNetGenerator, Discriminator, Latent_Discriminator, Discriminator, DiscriminatorWithClassifAttr, DiscriminatorWithMatchingAttr
from modules.perceptual_loss import PerceptualLoss, StyleLoss, VGG16FeatureExtractor
from modules.GAN_loss import GANLoss
from utils.im_util import denorm, write_labels_on_images
import numpy as np
from agents.trainingModule import TrainingModule



class FaderNet(TrainingModule):
    def __init__(self, config):
        super(FaderNet, self).__init__(config)

        self.norm='none'
        self.G = FaderNetGenerator(conv_dim=config.g_conv_dim,
                                   n_layers=config.g_layers,
                                   max_dim=config.max_conv_dim,
                                   im_channels=config.img_channels,
                                   skip_connections=config.skip_connections, 
                                   vgg_like=config.vgg_like, attr_dim=len(config.attrs),
                                   n_attr_deconv=config.n_attr_deconv,
                                   normalization=self.norm,
                                   first_conv=config.first_conv,
                                   n_bottlenecks=config.n_bottlenecks)

        if self.config.GAN_style == 'vanilla':
            self.D = Discriminator(image_size=config.image_size, im_channels=3, attr_dim=len(config.attrs), conv_dim=config.d_conv_dim,n_layers=config.d_layers,max_dim=config.max_conv_dim,fc_dim=config.d_fc_dim, normalization=self.norm)
        elif self.config.GAN_style == 'matching':
            self.D = DiscriminatorWithMatchingAttr(image_size=config.image_size, im_channels=3, attr_dim=len(config.attrs), conv_dim=config.d_conv_dim,n_layers=config.d_layers,max_dim=config.max_conv_dim,fc_dim=config.d_fc_dim, normalization=self.norm)
        elif self.config.GAN_style == 'classif':
            self.D = DiscriminatorWithClassifAttr(image_size=config.image_size, im_channels=3, attr_dim=len(config.attrs), conv_dim=config.d_conv_dim,n_layers=config.d_layers,max_dim=config.max_conv_dim,fc_dim=config.d_fc_dim, normalization=self.norm)
        self.LD = Latent_Discriminator(image_size=config.image_size, im_channels=config.img_channels, attr_dim=len(config.attrs), conv_dim=config.g_conv_dim,n_layers=config.g_layers,max_dim=config.max_conv_dim, fc_dim=config.d_fc_dim, skip_connections=config.skip_connections, vgg_like=config.vgg_like, normalization=self.norm, first_conv=config.first_conv)
        print(self.G)
        if self.config.use_image_disc:
            print(self.D)
        if self.config.use_latent_disc:
            print(self.LD)

         # create all the loss functions that we may need for perceptual loss
        self.loss_P = PerceptualLoss().to(self.device)
        self.loss_S = StyleLoss().to(self.device)
        self.vgg16_f = VGG16FeatureExtractor(['relu1_2', 'relu2_2', 'relu3_3', 'relu4_4']).to(self.device)
        if self.config.use_image_disc:
            self.criterionGAN = GANLoss(self.config.gan_mode).to(self.device)

        self.data_loader = globals()['{}_loader'.format(self.config.dataset)](
            self.config.data_root, self.config.mode, self.config.attrs,
            self.config.crop_size, self.config.image_size, self.config.batch_size, self.config.data_augmentation, mask_input_bg=config.mask_input_bg)

        self.logger.info("FaderNet ready")
        

    ################################################################
    ###################### SAVE/lOAD ###############################
    def save_checkpoint(self):
        self.save_one_model(self.G,self.optimizer_G,'G')
        if self.config.use_image_disc:
            self.save_one_model(self.D,self.optimizer_D,'D')
        if self.config.use_latent_disc:
            self.save_one_model(self.LD,self.optimizer_LD,'LD')

    def load_checkpoint(self):
        if self.config.checkpoint is None:
            return

        self.load_one_model(self.G,self.optimizer_G if self.config.mode=='train' else None,'G')
        if (self.config.use_image_disc and self.config.mode=='train'):
            self.load_one_model(self.D,self.optimizer_D if self.config.mode=='train' else None,'D')
        if self.config.use_latent_disc and self.config.mode=='train':
            self.load_one_model(self.LD,self.optimizer_LD if self.config.mode=='train' else None,'LD')

        self.current_iteration = self.config.checkpoint


    ################################################################
    ################### OPTIM UTILITIES ############################

    def setup_all_optimizers(self):
        self.optimizer_G = self.build_optimizer(self.G, self.config.g_lr)
        self.optimizer_D = self.build_optimizer(self.D, self.config.d_lr)
        self.optimizer_LD = self.build_optimizer(self.LD, self.config.ld_lr)
        self.load_checkpoint() #load checkpoint if needed 
        self.lr_scheduler_G = self.build_scheduler(self.optimizer_G)
        self.lr_scheduler_D = self.build_scheduler(self.optimizer_D,not(self.config.use_image_disc))
        self.lr_scheduler_LD = self.build_scheduler(self.optimizer_LD, not self.config.use_latent_disc)

    def step_schedulers(self):
        self.lr_scheduler_G.step()
        self.lr_scheduler_D.step()
        self.lr_scheduler_LD.step()
        self.scalars['lr/g_lr'] = self.lr_scheduler_G.get_lr()[0]
        self.scalars['lr/ld_lr'] = self.lr_scheduler_LD.get_lr()[0]
        self.scalars['lr/d_lr'] = self.lr_scheduler_D.get_lr()[0]

    def eval_mode(self):
        self.G.eval()
        self.LD.eval()
        self.D.eval()
    def training_mode(self):
        self.G.train()
        self.LD.train()
        self.D.train()

    def parallel_GPU(self):
        self.G = self.to_multi_GPU(self.G)
        self.D = self.to_multi_GPU(self.D)
        self.LD = self.to_multi_GPU(self.LD)

    ################################################################
    ##################### EVAL UTILITIES ###########################
    def decode(self,bneck,encodings,att):
        return self.G.decode(att,bneck,encodings)
    def encode(self):
        return self.G.encode(self.batch_Ia)
    def forward(self,new_attr=None):
        if new_attr != None: self.current_att=new_attr
        else: self.current_att = self.batch_a_att
        encodings,z,_ = self.encode()
        return self.decode(z,encodings,self.current_att)



    def init_sample_grid(self):
        x_fake_list = [self.batch_Ia[:,:3]]
        return x_fake_list

    def create_interpolated_attr(self, c_org, selected_attrs=None,max_val=5.0):
        """Generate target domain labels for debugging and testing: linearly sample attribute. Contains a list for each attr"""
        all_lists=[]
        for i in range(len(selected_attrs)):
            c_trg_list = []#[c_org]
            alphas = [-max_val, -((max_val-1)/2.0+1), -1,-0.5,0,0.5,1,((max_val-1)/2.0+1), max_val]
            if max_val==1: alphas = linspace(-1,1,9) #[-1,-0.75,-0.5,0,0.5,1,]
            #alphas = np.linspace(-max_val, max_val, 10)
            for alpha in alphas:
                c_trg = c_org.clone()
                c_trg[:, i] = torch.full_like(c_trg[:, i],alpha) 
                c_trg_list.append(c_trg)
            all_lists.append(c_trg_list)
        return all_lists



    def compute_sample_grid(self,batch,max_val,path=None,writer=False):
        self.batch_Ia, self.batch_normals, self.batch_illum, self.batch_a_att = batch
        all_sample_list = self.create_interpolated_attr(self.batch_a_att, self.config.attrs,max_val=max_val)

        all_images=[]
        for c_sample_list in all_sample_list:
            x_fake_list = self.init_sample_grid()
            for c_trg_sample in c_sample_list:
                fake_image=self.forward(c_trg_sample)*self.batch_Ia[:,3:]
                fake_image=torch.cat([fake_image,self.batch_Ia[:,3:]],dim=1)
                #write_labels_on_images(fake_image,c_trg_sample)
                x_fake_list.append(fake_image)
            all_images.append(x_fake_list)
        # #interleave the images for each attribute
        # size = all_images[0][0].shape
        # x_fake_list = []
        # for col in range(len(all_images[0])):
        #     x_fake_list.append(torch.stack([image[col] for image in all_images], dim=1).view(len(all_images)*size[0],size[1],size[2],size[3]))
        x_concat = torch.cat(x_fake_list, dim=3)
        image = tvutils.make_grid(denorm(x_concat,device=self.device), nrow=1)
        if writer:
            self.writer.add_image('sample', image,self.current_iteration)
        if path:
            tvutils.save_image(image,path)

    ########################################################################################
    #####################                 TRAINING               ###########################
    def train_latent_discriminator(self):
        self.G.eval()
        self.LD.train()
        # compute disc loss on encoded image
        _,bneck,bn_list = self.encode()

        for _ in range(self.config.n_critic_ld):
            out_att = self.LD(bneck,bn_list)
            #classification loss
            ld_loss = self.regression_loss(out_att, self.batch_a_att)*self.config.lambda_LD
            # backward and optimize
            self.optimize(self.optimizer_LD,ld_loss)
            # summarize
            self.scalars['LD/loss'] = ld_loss.item()

    def train_GAN_discriminator(self):
        self.G.eval()
        self.D.train()
        for _ in range(self.config.n_critic):
            # fake image Ib_hat
            b_att =  torch.rand_like(self.batch_a_att)*2-1.0 
            Ib_hat = self.forward(b_att)
            if self.config.GAN_style == 'vanilla': #simple image GAN
                # real image is true
                out_disc_real = self.D(self.batch_Ia[:,:3])
                d_loss_adv_real = self.criterionGAN(out_disc_real, True)
                #fake image
                out_disc_fake = self.D(Ib_hat.detach())
                d_loss_adv_fake = self.criterionGAN(out_disc_fake, False)
                # compute loss for gradient penalty
                d_loss_adv_gp = GANLoss.cal_gradient_penalty(self.D,self.batch_Ia[:,:3],Ib_hat,self.device,lambda_gp=self.config.lambda_gp, attribute=self.batch_a_att)
                #full GAN loss
                d_loss = d_loss_adv_real + d_loss_adv_fake + d_loss_adv_gp
                self.scalars['D/loss_real'] = d_loss_adv_real.item()
                self.scalars['D/loss_fake'] = d_loss_adv_fake.item()
                self.scalars['D/loss_gp'] = d_loss_adv_gp.item()
            elif self.config.GAN_style == 'matching': #GAN with matching attr + real/fake
                # MATCHING real image with good attr is true
                out_disc_real, out_match_real = self.D(self.batch_Ia[:,:3], self.batch_a_att)
                d_loss_match_real = self.criterionGAN(out_match_real, True)
                # MATCHING real image with bad attr is fake
                _,out_match_fake_att = self.D(self.batch_Ia[:,:3],b_att)
                d_loss_match_fake_att = self.criterionGAN(out_match_fake_att, False)
                # real image is true
                d_loss_adv_real = self.criterionGAN(out_disc_real, True)
                # MATCHING fake image  with its attr is fake
                out_disc_fake, out_match_fake_img = self.D(Ib_hat.detach(),b_att)
                d_loss_match_fake_img = self.criterionGAN(out_match_fake_img, False)
                # fake image is false
                d_loss_adv_fake = self.criterionGAN(out_disc_fake, False)
                # compute loss for gradient penalty
                d_loss_adv_gp = GANLoss.cal_gradient_penalty(self.D,self.batch_Ia[:,:3],Ib_hat, self.batch_a_att,self.device,lambda_gp=self.config.lambda_gp, attribute=self.batch_a_att)
                #full GAN loss
                d_loss_adv=d_loss_adv_real + d_loss_adv_fake + d_loss_adv_gp
                d_loss_match = d_loss_match_real + 0.5*d_loss_match_fake_att + 0.5*d_loss_match_fake_img
                d_loss = self.config.d_lamdba_adv * d_loss_adv + self.config.d_lambda_match * d_loss_match
                self.scalars['D/loss_match_real'] = d_loss_match_real.item()
                self.scalars['D/loss_match_fake_att'] = d_loss_match_fake_att.item()
                self.scalars['D/loss_match_fake_img'] = d_loss_match_fake_img.item()
                self.scalars['D/loss_real'] = d_loss_adv_real.item()
                self.scalars['D/loss_fake'] = d_loss_adv_fake.item()
                self.scalars['D/loss_gp'] = d_loss_adv_gp.item()
            elif self.config.GAN_style == 'classif': #GAN with classif attr + real/fake
                # real image is true
                out_disc_real, out_classif = self.D(self.batch_Ia[:,:3])
                d_loss_adv_real = self.criterionGAN(out_disc_real, True)
                #learn to classify
                d_loss_classif = self.config.d_lambda_classif * self.regression_loss(out_classif,self.batch_a_att)
                #fake image
                out_disc_fake, _ = self.D(Ib_hat.detach())
                d_loss_adv_fake = self.criterionGAN(out_disc_fake, False)
                # compute loss for gradient penalty
                d_loss_adv_gp = GANLoss.cal_gradient_penalty(self.D,self.batch_Ia[:,:3],Ib_hat,self.device,lambda_gp=self.config.lambda_gp, attribute=self.batch_a_att)
                #full GAN loss
                d_loss_adv = d_loss_adv_real + d_loss_adv_fake + d_loss_adv_gp
                d_loss = self.config.d_lamdba_adv * d_loss_adv + d_loss_classif
                self.scalars['D/loss_real'] = d_loss_adv_real.item()
                self.scalars['D/loss_fake'] = d_loss_adv_fake.item()
                self.scalars['D/loss_gp'] = d_loss_adv_gp.item()
                self.scalars['D/loss_classif'] = d_loss_classif.item()

            # backward and optimize
            self.optimize(self.optimizer_D,d_loss)
            # summarize
            self.scalars['D/loss'] = d_loss.item()

    def training_step(self, batch):
        self.batch_Ia, self.batch_normals, self.batch_illum, self.batch_a_att = batch

        # ================================================================================= #
        #                           2. Train the discriminator                              #
        # ================================================================================= #
        if self.config.use_image_disc:
            self.train_GAN_discriminator()

        # ================================================================================= #
        #                        3. Train the latent discriminator (FaderNet)               #
        # ================================================================================= #
        if self.config.use_latent_disc:
            self.train_latent_discriminator()

        # ================================================================================= #
        #                              3. Train the generator                               #
        # ================================================================================= #
        self.G.train()
        self.D.eval()
        self.LD.eval()

        self.current_att=self.batch_a_att       

        encodings,bneck,bn_list = self.encode()
        Ia_hat=self.decode(bneck,encodings,self.batch_a_att)

        #reconstruction loss
        g_loss_rec = self.config.lambda_G_rec * self.image_reconstruction_loss(self.batch_Ia[:,:3],Ia_hat)
        g_loss = g_loss_rec
        self.scalars['G/loss_rec'] = g_loss_rec.item()
        #tv loss
        if self.config.lambda_G_tv > 0:
            g_tv_loss = self.config.lambda_G_tv * (
                torch.mean(torch.abs(Ia_hat[:, :, :, :-1] - Ia_hat[:, :, :, 1:])) + 
                torch.mean(torch.abs(Ia_hat[:, :, :-1, :] - Ia_hat[:, :, 1:, :]))
            )
            g_loss += g_tv_loss
            self.scalars['G/loss_tv'] = g_tv_loss.item()
        
        #latent discriminator for attribute
        if self.config.use_latent_disc:
            out_att = self.LD(bneck,bn_list)
            g_loss_latent = -self.config.lambda_G_latent * self.regression_loss(out_att, self.batch_a_att)
            g_loss += g_loss_latent
            self.scalars['G/loss_latent'] = g_loss_latent.item()

        if self.config.use_image_disc:
            b_att =  torch.rand_like(self.batch_a_att)*2-1.0 
            # original-to-target domain : Ib_hat -> GAN + classif
            Ib_hat = self.forward(b_att)
            if self.config.GAN_style == 'vanilla':
                out_disc = self.D(Ib_hat)
                loss_adv=self.config.lambda_adv*self.criterionGAN(out_disc, True)
                g_loss_adv = loss_adv
            elif self.config.GAN_style == 'matching':
                out_disc, out_match = self.D(Ib_hat,b_att)
                loss_adv=self.config.lambda_adv*self.criterionGAN(out_disc, True)
                loss_match = self.config.lambda_adv*self.criterionGAN(out_match, True)
                g_loss_adv = loss_adv + loss_match
                self.scalars['G/loss_adv_match'] = loss_match.item()
            elif self.config.GAN_style == 'classif':
                out_disc, out_classif = self.D(Ib_hat)
                loss_adv=self.config.lambda_adv*self.criterionGAN(out_disc, True)
                loss_classif = self.config.lambda_adv_classif*self.regression_loss(out_classif, b_att)
                g_loss_adv = loss_adv + loss_classif
                self.scalars['G/loss_adv_classif'] = loss_classif.item()
            # GAN loss
            g_loss += g_loss_adv
            self.scalars['G/loss_adv_img'] = loss_adv.item()
            self.scalars['G/loss_adv'] = g_loss_adv.item()

        # backward and optimize
        self.optimize(self.optimizer_G,g_loss)
        # summarize
        self.scalars['G/loss'] = g_loss.item()
        return self.scalars

    def validating_step(self, batch):
        self.compute_sample_grid(batch,5.0,os.path.join(self.config.sample_dir, 'sample_{}.png'.format(self.current_iteration)),writer=True)


    def testing_step(self, batch, batch_id):
        i=batch_id
        self.compute_sample_grid(batch,1.0,os.path.join(self.config.result_dir, "sample_{}_{}.png".format(i + 1,self.config.checkpoint)),writer=False)
        #self.compute_sample_grid(batch,5.0,os.path.join(self.config.result_dir, 'sample_big_{}_{}.png'.format(i + 1,self.config.checkpoint)),writer=False)
        #self.output_results(batch,i)



    def output_results(self,batch,batch_id):
        self.batch_Ia, self.batch_normals, filename, self.batch_a_att = batch
        all_sample_list = self.create_interpolated_attr(self.batch_a_att, self.config.attrs,max_val=1)
        path=os.path.join(self.config.result_dir,str(self.config.checkpoint),"test")
        os.makedirs(path,exist_ok=True)
        batch_size=32

        all_images=[]
        for c_sample_list in all_sample_list: #for each attr
            for c_trg_sample in c_sample_list:
                #print(c_trg_sample)
                fake_image=self.forward(c_trg_sample)*self.batch_Ia[:,3:]
                fake_image=denorm(fake_image,device=self.device)
                fake_image=torch.cat([fake_image,self.batch_Ia[:,3:]],dim=1)
                #print(fake_image.shape)
                #write_labels_on_images(fake_image,c_trg_sample)
                for i in range(fake_image.shape[0]):
                    attr_string=str(c_trg_sample[i].item()).replace('.','_')
                    tvutils.save_image(fake_image[i],os.path.join(path, "{}_{}_{}.png".format(filename[i],self.config.attrs[0],attr_string)))

