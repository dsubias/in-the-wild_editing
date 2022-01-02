# OS imports
import os
import logging
import re

# Tensor imports
from numpy.core.function_base import linspace
import torch
import torch.nn.functional as F
import torchvision.utils as tvutils
import pytorch_lightning as pl
import torch.optim as optim

# src imports
from datasets import *
from models.networks import *
from modules.perceptual_loss import PerceptualLoss, StyleLoss, VGG16FeatureExtractor
from modules.GAN_loss import GANLoss
from utils.im_util import denorm

logger = logging.getLogger('exp')


class FaderNetPL(pl.LightningModule):

    def __init__(self, config):

        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.norm = 'none'
        self.automatic_optimization = False
        self.current_iteration = 0

        # Generator definition
        if self.config.network == 'FaderNetWithNormals':

            logger.info("Preparing Generator 1 [G1]...")

            self.G = FaderNetGeneratorWithNormals(conv_dim=config.g_conv_dim,
                                                  n_layers=config.g_layers,
                                                  max_dim=config.max_conv_dim,
                                                  im_channels=config.img_channels,
                                                  skip_connections=config.skip_connections,
                                                  vgg_like=config.vgg_like,
                                                  attr_dim=len(config.attrs),
                                                  n_attr_deconv=config.n_attr_deconv,
                                                  n_concat_normals=config.n_concat_normals,
                                                  normalization=self.norm,
                                                  first_conv=config.first_conv,
                                                  n_bottlenecks=config.n_bottlenecks,
                                                  img_size=self.config.image_size,
                                                  batch_size=self.config.batch_size)

        elif self.config.network == 'FaderNetWithNormals2Steps':

            logger.info("Preparing Generator 2 [G2]...")
            self.G = FaderNetGeneratorWithNormals2Steps(conv_dim=config.g_conv_dim,
                                                        n_layers=config.g_layers,
                                                        max_dim=config.max_conv_dim,
                                                        im_channels=config.img_channels,
                                                        skip_connections=config.skip_connections,
                                                        vgg_like=config.vgg_like,
                                                        attr_dim=len(
                                                            config.attrs),
                                                        n_attr_deconv=config.n_attr_deconv,
                                                        n_concat_normals=config.n_concat_normals,
                                                        normalization=self.norm,
                                                        first_conv=config.first_conv,
                                                        n_bottlenecks=config.n_bottlenecks,
                                                        all_feat=config.all_feat,
                                                        img_size=self.config.image_size,
                                                        batch_size=self.config.batch_size,
                                                        device=self.device)

            self.G_small = FaderNetGeneratorWithNormals(conv_dim=32,
                                                        n_layers=6,
                                                        max_dim=512,
                                                        im_channels=config.img_channels,
                                                        skip_connections=0,
                                                        vgg_like=0,
                                                        attr_dim=len(
                                                            config.attrs),
                                                        n_attr_deconv=1,
                                                        n_concat_normals=4,
                                                        normalization=self.norm,
                                                        first_conv=False,
                                                        n_bottlenecks=2)
            # Load de pretrained generator 1
            self.load_generator(self.G_small, config.faderNet_checkpoint)
            self.G_small.eval()
            self.G_small = self.to_multi_GPU(self.G_small)

        elif self.config.network == 'FaderNet':

            logger.info("Preparing simple Fader Net...")
            self.G = FaderNetGenerator(conv_dim=config.g_conv_dim,
                                       n_layers=config.g_layers,
                                       max_dim=config.max_conv_dim,
                                       im_channels=config.img_channels,
                                       skip_connections=config.skip_connections,
                                       vgg_like=config.vgg_like,
                                       attr_dim=len(config.attrs),
                                       n_attr_deconv=config.n_attr_deconv,
                                       normalization=self.norm,
                                       first_conv=config.first_conv,
                                       n_bottlenecks=config.n_bottlenecks)

            if self.config.skip_connections and self.config.use_LLD:
                logger.info("Preparing Layer Discriminators...")
                self.shift = 0 if not self.config.first_conv else 1
                config.skip_connections = min(
                    self.config.skip_connections, self.config.g_layers - 1)
                for i in range(self.shift, self.config.skip_connections):
                    output = min(2 ** i * config.g_conv_dim,
                                 config.max_conv_dim)
                    setattr(self, 'LD_layer_{}'.format(i), Latent_Discriminator_Layer(i,
                                                                                      image_size=config.image_size,
                                                                                      im_channels=output,
                                                                                      attr_dim=len(
                                                                                          config.attrs),
                                                                                      conv_dim=config.g_conv_dim,
                                                                                      n_layers=config.g_layers,
                                                                                      max_dim=config.max_conv_dim,
                                                                                      fc_dim=config.d_fc_dim,
                                                                                      skip_connections=config.skip_connections,
                                                                                      vgg_like=config.vgg_like,
                                                                                      normalization=self.norm,
                                                                                      first_conv=config.first_conv))

        else:
            logger.info("Model no available")

        if self.config.GAN_style == 'vanilla':
            self.D = Discriminator(image_size=config.image_size, im_channels=3, attr_dim=len(config.attrs), conv_dim=config.d_conv_dim,
                                   n_layers=config.d_layers, max_dim=config.max_conv_dim, fc_dim=config.d_fc_dim, normalization=self.norm)
        elif self.config.GAN_style == 'matching':
            self.D = DiscriminatorWithMatchingAttr(image_size=config.image_size, im_channels=3, attr_dim=len(
                config.attrs), conv_dim=config.d_conv_dim, n_layers=config.d_layers, max_dim=config.max_conv_dim, fc_dim=config.d_fc_dim, normalization=self.norm)
        elif self.config.GAN_style == 'classif':
            self.D = DiscriminatorWithClassifAttr(image_size=config.image_size, im_channels=3, attr_dim=len(
                config.attrs), conv_dim=config.d_conv_dim, n_layers=config.d_layers, max_dim=config.max_conv_dim, fc_dim=config.d_fc_dim, normalization=self.norm)
        self.LD = Latent_Discriminator(image_size=config.image_size, im_channels=config.img_channels, attr_dim=len(config.attrs), conv_dim=config.g_conv_dim, n_layers=config.g_layers,
                                       max_dim=config.max_conv_dim, fc_dim=config.d_fc_dim, skip_connections=config.skip_connections, vgg_like=config.vgg_like, normalization=self.norm, first_conv=config.first_conv)

        # create all the loss functions that we may need for perceptual loss
        self.loss_P = PerceptualLoss().to(self.device)
        self.loss_S = StyleLoss().to(self.device)
        self.vgg16_f = VGG16FeatureExtractor(
            ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_4']).to(self.device)

        if self.config.use_image_disc:
            self.criterionGAN = GANLoss(self.config.gan_mode).to(self.device)

        self.scalars = {}
        self.scalars['lr/g_lr'] = []
        self.scalars['lr/d_lr'] = []
        self.scalars['lr/ld_lr'] = []
        for i in range(self.shift, self.config.skip_connections):
            self.scalars['LD_layer_{}/loss'.format(i)] = []
            self.scalars['G/loss_latent_{}'.format(i)] = []
            self.scalars['lr/ld_{}_lr'.format(i)] = []

        self.scalars['LD/loss'] = []
        self.scalars['D/loss_real'] = []
        self.scalars['D/loss_fake'] = []
        self.scalars['D/loss_gp'] = []
        self.scalars['D/loss_match_real'] = []
        self.scalars['D/loss_match_fake_att'] = []
        self.scalars['D/loss_match_fake_img'] = []
        self.scalars['D/loss_real'] = []
        self.scalars['D/loss_fake'] = []
        self.scalars['D/loss_gp'] = []
        self.scalars['D/loss_real'] = []
        self.scalars['D/loss_fake'] = []
        self.scalars['D/loss_gp'] = []
        self.scalars['D/loss_classif'] = []
        self.scalars['D/loss'] = []
        self.scalars['G/loss_rec'] = []
        self.scalars['G/loss_tv'] = []
        self.scalars['G/loss_latent'] = []
        self.scalars['G/loss_adv_match'] = []
        self.scalars['G/loss_adv_classif'] = []
        self.scalars['G/loss_adv_img'] = []
        self.scalars['G/loss_adv'] = []
        self.scalars['G/loss'] = []
        self.scalars['G/loss'] = []
        self.scalars['G/loss_rec_l1'] = []
        self.scalars['G/loss_rec_perc'] = []
        self.scalars['G/loss_rec_style'] = []

        logger.info("Generator ready")

    # ================================================================================= #
    #                                  CONFIGURATION                                    #
    # ================================================================================= #

    def eval_mode(self):

        self.G.eval()
        self.LD.eval()
        self.D.eval()
        if self.config.skip_connections and self.config.use_LLD:
            for i in range(self.shift, self.config.skip_connections):
                getattr(self, 'LD_layer_{}'.format(i)).eval()

    def training_mode(self):

        self.G.train()
        self.LD.train()
        self.D.train()
        if self.config.skip_connections and self.config.use_LLD:
            for i in range(self.shift, self.config.skip_connections):
                getattr(self, 'LD_layer_{}'.format(i)).train()

    def parallel_GPU(self):

        self.G = self.to_multi_GPU(self.G)
        self.D = self.to_multi_GPU(self.D)
        self.LD = self.to_multi_GPU(self.LD)

    # ================================================================================= #
    #                                     SAVE/lOAD                                     #
    # ================================================================================= #

    def save_checkpoint(self):
        self.save_one_model(self.G, self.optimizer_G, 'G')
        if self.config.use_image_disc:
            self.save_one_model(self.D, self.optimizer_D, 'D')
        if self.config.use_latent_disc:
            self.save_one_model(self.LD, self.optimizer_LD, 'LD')

    def load_checkpoint(self):

        if self.config.checkpoint is None:
            return

        self.load_one_model(
            self.G, self.optimizer_G if self.config.mode == 'train' else None, 'G')
        if (self.config.use_image_disc and self.config.mode == 'train'):
            self.load_one_model(
                self.D, self.optimizer_D if self.config.mode == 'train' else None, 'D')
        if self.config.use_latent_disc and self.config.mode == 'train':
            self.load_one_model(
                self.LD, self.optimizer_LD if self.config.mode == 'train' else None, 'LD')

        self.current_iteration = self.config.checkpoint

    def load_generator(self, model, path):

        checkpoint = torch.load(path, map_location=self.device)
        to_load = {}

        # Choosing only generator params
        for k, v in checkpoint['state_dict'].items():
            if re.search(r"^G.+$", k):
                to_load[k.replace('G.', '')] = v

        model.load_state_dict(to_load)
        model.to(self.device)

    # ================================================================================= #
    #                                    OPTIM UTILITIES                                #
    # ================================================================================= #

    def build_scheduler(self, optimizer, not_load=False):

        if (self.config.checkpoint == None or not_load):
            last_epoch = -1
        else:
            last_epoch = self.config.checkpoint

        return optim.lr_scheduler.StepLR(optimizer,
                                         step_size=self.config.lr_decay_iters,
                                         gamma=0.5,
                                         last_epoch=last_epoch)

    def optimize(self, optimizer, loss):

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    def configure_optimizers(self):

        optimizer_G = optim.Adam(self.G.parameters(),  self.config.g_lr, [
                                 self.config.beta1, self.config.beta2])
        optimizer_D = optim.Adam(self.D.parameters(),  self.config.d_lr, [
                                 self.config.beta1, self.config.beta2])
        optimizer_LD = optim.Adam(self.LD.parameters(),  self.config.ld_lr, [
                                  self.config.beta1, self.config.beta2])
        layers_optmimizers = []
        if self.config.skip_connections and self.config.use_LLD:
            for i in range(self.shift, self.config.skip_connections):
                layers_optmimizers.append(optim.Adam(getattr(self, 'LD_layer_{}'.format(i)).parameters(),  self.config.ld_lr, [
                    self.config.beta1, self.config.beta2]))

        self.load_checkpoint()  # load checkpoint if needed
        lr_scheduler_G = self.build_scheduler(optimizer_G)
        lr_scheduler_D = self.build_scheduler(
            optimizer_D, not(self.config.use_image_disc))
        lr_scheduler_LD = self.build_scheduler(
            optimizer_LD, not self.config.use_latent_disc)

        layers_scheduler = []
        if self.config.skip_connections and self.config.use_LLD:
            for i in range(self.shift, self.config.skip_connections):
                layers_scheduler.append(self.build_scheduler(
                    layers_optmimizers[i], not self.config.use_latent_disc))

        return [optimizer_G, optimizer_D, optimizer_LD] + layers_optmimizers, [lr_scheduler_G, lr_scheduler_D, lr_scheduler_LD] + layers_scheduler

    def step_schedulers(self):

        schedulers = self.lr_schedulers()

        for i in range(len(schedulers)):
            schedulers[i].step()

        self.scalars['lr/g_lr'].append(schedulers[0].get_last_lr()[0])
        self.scalars['lr/d_lr'].append(schedulers[1].get_last_lr()[0])
        self.scalars['lr/ld_lr'].append(schedulers[2].get_last_lr()[0])

        if self.config.skip_connections and self.config.use_LLD:

            for i in range(0, len(schedulers)-4):
                self.scalars['lr/ld_{}_lr'.format(i)
                             ].append(schedulers[i].get_last_lr()[0])

    # ================================================================================= #
    #                                   EVAL UTILITIES                                  #
    # ================================================================================= #

    def decode(self, bneck, encodings, att):

        # Decoding of the generator 1
        if self.config.network == 'FaderNetWithNormals':

            normals = self.get_normals()
            return self.G.decode(att, bneck, normals, encodings)

        # Decoding of the generator 2
        elif self.config.network == 'FaderNetWithNormals2Steps':

            normals = self.get_normals()
            fn_output, fn_features, _ = self.get_fadernet_output(att)
            return self.G.decode(att, bneck, normals, fn_features, encodings)

        # Decoding of simple fader network
        return self.G.decode(att, bneck, encodings)

    def encode(self):
        return self.G.encode(self.batch_Ia)

    def forward(self, new_attr=None):

        if new_attr != None:
            self.current_att = new_attr
        else:
            self.current_att = self.batch_a_att

        encodings, z, _ = self.encode()
        return self.decode(z, encodings, self.current_att)

    def init_sample_grid(self):
        x_fake_list = [self.batch_normals, self.batch_Ia]
        return x_fake_list

    def init_sample_grid_2(self, batch_normals, batch_Ia):
        x_fake_list = [batch_normals, batch_Ia]
        return x_fake_list

    def create_interpolated_attr(self, c_org, selected_attrs=None, max_val=5.0):
        """Generate target domain labels for debugging and testing: linearly sample attribute. Contains a list for each attr"""
        all_lists = []
        for i in range(len(selected_attrs)):
            c_trg_list = []  # [c_org]
            alphas = [-max_val, -((max_val-1)/2.0+1), -1, -
                      0.5, 0, 0.5, 1, ((max_val-1)/2.0+1), max_val]
            if max_val == 1:
                alphas = linspace(-1, 1, 9)  # [-1,-0.75,-0.5,0,0.5,1,]
            #alphas = np.linspace(-max_val, max_val, 10)
            for alpha in alphas:
                c_trg = c_org.clone()
                c_trg[:, i] = torch.full_like(c_trg[:, i], alpha)
                c_trg_list.append(c_trg)
            all_lists.append(c_trg_list)
        return all_lists

    def compute_sample_grid(self, batch, max_val, path, plot_features=0):

        self.batch_Ia, self.batch_normals, self.batch_illum, self.batch_a_att = batch
        all_sample_list = self.create_interpolated_attr(
            self.batch_a_att, self.config.attrs, max_val=max_val)
        all_images = []
        for c_sample_list in all_sample_list:

            x_fake_list = self.init_sample_grid()

            for c_trg_sample in c_sample_list:

                fake_image = self.forward(c_trg_sample)*self.batch_Ia[:, 3:]
                fake_image = torch.cat(
                    [fake_image, self.batch_Ia[:, 3:]], dim=1)
                # write_labels_on_images(fake_image,c_trg_sample)
                x_fake_list.append(fake_image)

            all_images.append(x_fake_list)

        x_concat = torch.cat(x_fake_list, dim=3)
        image = tvutils.make_grid(denorm(x_concat, device=self.device), nrow=1)
        tvutils.save_image(image, path)

        if plot_features:
            encodings, bneck, bn_list = self.encode()
            for i in range(encodings[plot_features].shape[1]):
                tvutils.save_image(
                    encodings[plot_features][1, i, :], 'features/features_{}.png'.format(i))

    # ================================================================================= #
    #                                     TRAINING                                      #
    # ================================================================================= #

    def train_layer_discriminators(self, optimizers):

        self.G.eval()
        for i in range(self.shift, self.config.skip_connections):
            getattr(self, 'LD_layer_{}'.format(i)).train()
        # compute disc loss on encoded image
        encodings, bneck, bn_list = self.encode()

        for i in range(self.shift, self.config.skip_connections):
            for _ in range(self.config.n_critic_ld):
                out_att = getattr(self, 'LD_layer_{}'.format(i))(
                    encodings[i], bn_list)

                # classification loss
                ld_loss = self.regression_loss(
                    out_att, self.batch_a_att)*self.config.lambda_LD
                # backward and optimize
                self.optimize(optimizers[i], ld_loss)
                # summarize
                self.scalars['LD_layer_{}/loss'.format(i)
                             ].append(ld_loss.item())

    def train_latent_discriminator(self, optimizer_LD):

        self.G.eval()
        self.LD.train()

        # compute disc loss on encoded image
        _, bneck, bn_list = self.encode()

        for _ in range(self.config.n_critic_ld):

            out_att = self.LD(bneck, bn_list)

            # classification loss
            ld_loss = self.regression_loss(
                out_att, self.batch_a_att)*self.config.lambda_LD
            # backward and optimize
            self.optimize(optimizer_LD, ld_loss)
            # summarize
            self.scalars['LD/loss'].append(ld_loss.item())

    def train_GAN_discriminator(self, optimizer_D):

        self.G.eval()
        self.D.train()

        for _ in range(self.config.n_critic):

            # fake image Ib_hat
            b_att = torch.rand_like(self.batch_a_att)*2-1.0
            Ib_hat = self.forward(b_att)
            if self.config.GAN_style == 'vanilla':  # simple image GAN

                # real image is true
                out_disc_real = self.D(self.batch_Ia[:, :3])
                d_loss_adv_real = self.criterionGAN(out_disc_real, True)
                # fake image
                out_disc_fake = self.D(Ib_hat.detach())
                d_loss_adv_fake = self.criterionGAN(out_disc_fake, False)
                # compute loss for gradient penalty
                d_loss_adv_gp = GANLoss.cal_gradient_penalty(
                    self.D, self.batch_Ia[:, :3],
                    Ib_hat, self.device,
                    lambda_gp=self.config.lambda_gp,
                    attribute=self.batch_a_att)
                # full GAN loss
                d_loss = d_loss_adv_real + d_loss_adv_fake + d_loss_adv_gp
                self.scalars['D/loss_real'].append(d_loss_adv_real.item())
                self.scalars['D/loss_fake'].append(d_loss_adv_fake.item())
                self.scalars['D/loss_gp'].append(d_loss_adv_gp.item())

            elif self.config.GAN_style == 'matching':  # GAN with matching attr + real/fake

                # MATCHING real image with good attr is true
                out_disc_real, out_match_real = self.D(
                    self.batch_Ia[:, :3], self.batch_a_att)
                d_loss_match_real = self.criterionGAN(out_match_real, True)
                # MATCHING real image with bad attr is fake
                _, out_match_fake_att = self.D(self.batch_Ia[:, :3], b_att)
                d_loss_match_fake_att = self.criterionGAN(
                    out_match_fake_att, False)
                # real image is true
                d_loss_adv_real = self.criterionGAN(out_disc_real, True)
                # MATCHING fake image  with its attr is fake
                out_disc_fake, out_match_fake_img = self.D(
                    Ib_hat.detach(), b_att)
                d_loss_match_fake_img = self.criterionGAN(
                    out_match_fake_img, False)
                # fake image is false
                d_loss_adv_fake = self.criterionGAN(out_disc_fake, False)
                # compute loss for gradient penalty
                d_loss_adv_gp = GANLoss.cal_gradient_penalty(
                    self.D, self.batch_Ia[:, :3],
                    Ib_hat,
                    self.batch_a_att,
                    self.device,
                    lambda_gp=self.config.lambda_gp,
                    attribute=self.batch_a_att)

                # full GAN loss
                d_loss_adv = d_loss_adv_real + d_loss_adv_fake + d_loss_adv_gp
                d_loss_match = d_loss_match_real + 0.5 * \
                    d_loss_match_fake_att + 0.5*d_loss_match_fake_img
                d_loss = self.config.d_lamdba_adv * d_loss_adv + \
                    self.config.d_lambda_match * d_loss_match
                self.scalars['D/loss_match_real'].append(
                    d_loss_match_real.item())
                self.scalars['D/loss_match_fake_att'].append(
                    d_loss_match_fake_att.item())
                self.scalars['D/loss_match_fake_img'].append(
                    d_loss_match_fake_img.item())
                self.scalars['D/loss_real'].append(d_loss_adv_real.item())
                self.scalars['D/loss_fake'].append(d_loss_adv_fake.item())
                self.scalars['D/loss_gp'].append(d_loss_adv_gp.item())

            elif self.config.GAN_style == 'classif':  # GAN with classif attr + real/fake

                # real image is true
                out_disc_real, out_classif = self.D(self.batch_Ia[:, :3])
                d_loss_adv_real = self.criterionGAN(out_disc_real, True)
                # learn to classify
                d_loss_classif = self.config.d_lambda_classif * \
                    self.regression_loss(out_classif, self.batch_a_att)
                # fake image
                out_disc_fake, _ = self.D(Ib_hat.detach())
                d_loss_adv_fake = self.criterionGAN(out_disc_fake, False)
                # compute loss for gradient penalty
                d_loss_adv_gp = GANLoss.cal_gradient_penalty(
                    self.D, self.batch_Ia[:, :3], Ib_hat, self.device, lambda_gp=self.config.lambda_gp, attribute=self.batch_a_att)
                # full GAN loss
                d_loss_adv = d_loss_adv_real + d_loss_adv_fake + d_loss_adv_gp
                d_loss = self.config.d_lamdba_adv * d_loss_adv + d_loss_classif
                self.scalars['D/loss_real'].append(d_loss_adv_real.item())
                self.scalars['D/loss_fake'].append(d_loss_adv_fake.item())
                self.scalars['D/loss_gp'].append(d_loss_adv_gp.item())
                self.scalars['D/loss_classif'].append(d_loss_classif.item())

            # backward and optimize
            self.optimize(optimizer_D, d_loss)
            # summarize
            self.scalars['D/loss'].append(d_loss.item())

    def get_normals(self):

        return self.batch_normals[:, :3]

    def training_step(self, batch, batch_idx):

        self.training_mode()
        self.batch_Ia, self.batch_normals, self.batch_illum, self.batch_a_att = batch
        optimizers = self.optimizers()

        # ================================================================================= #
        #                           2. Train the discriminator                              #
        # ================================================================================= #

        if self.config.use_image_disc:
            self.train_GAN_discriminator(optimizers[1])

        # ================================================================================= #
        #                        3. Train the layer discriminators (FaderNet)               #
        # ================================================================================= #

        if self.config.skip_connections and self.config.use_LLD:

            self.train_layer_discriminators(optimizers[3:])

        # ================================================================================= #
        #                        4. Train the latent discriminator (FaderNet)               #
        # ================================================================================= #

        if self.config.use_latent_disc:
            self.train_latent_discriminator(optimizers[2])

        # ================================================================================= #
        #                              5. Train the generator                               #
        # ================================================================================= #

        self.G.train()
        self.D.eval()
        self.LD.eval()
        if self.config.skip_connections and self.config.use_LLD:
            for i in range(self.shift, self.config.skip_connections):
                getattr(self, 'LD_layer_{}'.format(i)).eval()

        self.current_att = self.batch_a_att

        encodings, bneck, bn_list = self.encode()

        Ia_hat = self.decode(bneck, encodings, self.batch_a_att)

        # reconstruction loss
        g_loss_rec = self.config.lambda_G_rec * \
            self.image_reconstruction_loss(self.batch_Ia[:, :3], Ia_hat)
        g_loss = g_loss_rec
        self.scalars['G/loss_rec'].append(g_loss_rec.item())

        # tv loss
        if self.config.lambda_G_tv > 0:
            g_tv_loss = self.config.lambda_G_tv * (
                torch.mean(torch.abs(Ia_hat[:, :, :, :-1] - Ia_hat[:, :, :, 1:])) +
                torch.mean(
                    torch.abs(Ia_hat[:, :, :-1, :] - Ia_hat[:, :, 1:, :]))
            )
            g_loss += g_tv_loss
            self.scalars['G/loss_tv'].append(g_tv_loss.item())

        # latent discriminator for attribute
        if self.config.use_latent_disc:
            # Compute the loss for layer discriminator
            if self.config.skip_connections and self.config.use_LLD:
                for i in range(self.shift, self.config.skip_connections):
                    out_att = getattr(self, 'LD_layer_{}'.format(i))(
                        encodings[i], bn_list)
                    g_loss_latent = -self.config.lambda_G_latent_layer * \
                        self.regression_loss(out_att, self.batch_a_att)
                    g_loss += g_loss_latent
                    self.scalars['G/loss_latent_{}'.format(i)
                                 ].append(g_loss_latent.item())

            out_att = self.LD(bneck, bn_list)
            g_loss_latent = -self.config.lambda_G_latent * \
                self.regression_loss(out_att, self.batch_a_att)
            g_loss += g_loss_latent
            self.scalars['G/loss_latent'].append(g_loss_latent.item())

        if self.config.use_image_disc:
            b_att = torch.rand_like(self.batch_a_att)*2-1.0
            # original-to-target domain : Ib_hat -> GAN + classif
            Ib_hat = self.forward(b_att)
            if self.config.GAN_style == 'vanilla':
                out_disc = self.D(Ib_hat)
                loss_adv = self.config.lambda_adv * \
                    self.criterionGAN(out_disc, True)
                g_loss_adv = loss_adv
            elif self.config.GAN_style == 'matching':
                out_disc, out_match = self.D(Ib_hat, b_att)
                loss_adv = self.config.lambda_adv * \
                    self.criterionGAN(out_disc, True)
                loss_match = self.config.lambda_adv * \
                    self.criterionGAN(out_match, True)
                g_loss_adv = loss_adv + loss_match
                self.scalars['G/loss_adv_match'].append(loss_match.item())
            elif self.config.GAN_style == 'classif':
                out_disc, out_classif = self.D(Ib_hat)
                loss_adv = self.config.lambda_adv * \
                    self.criterionGAN(out_disc, True)
                loss_classif = self.config.lambda_adv_classif * \
                    self.regression_loss(out_classif, b_att)
                g_loss_adv = loss_adv + loss_classif
                self.scalars['G/loss_adv_classif'].append(loss_classif.item())
            # GAN loss
            g_loss += g_loss_adv
            self.scalars['G/loss_adv_img'].append(loss_adv.item())
            self.scalars['G/loss_adv'].append(g_loss_adv.item())

        # backward and optimize
        self.optimize(optimizers[0], g_loss)
        self.step_schedulers()

        # summarize
        self.scalars['G/loss'].append(g_loss.item())

        # print summary on terminal and on tensorboard
        if self.current_iteration % self.config.summary_step == 0:
            for tag, value in self.scalars.items():

                if len(value) != 0:
                    self.log(tag, sum(value) / float(len(value)))
                    self.scalars[tag] = []

        self.current_iteration += 1
        return self.scalars

    def validation_step(self, batch, batch_idx):
        self.eval_mode()
        max_val = 5.0
        path = os.path.join(self.config.sample_dir,
                            'validation_sample_{}.png'.format(
                                self.current_iteration))

        self.batch_Ia, self.batch_normals, self.batch_illum, self.batch_a_att = batch

        all_sample_list = self.create_interpolated_attr(
            self.batch_a_att, self.config.attrs, max_val=max_val)

        all_images = []
        for c_sample_list in all_sample_list:

            x_fake_list = self.init_sample_grid()

            for c_trg_sample in c_sample_list:

                fake_image = self.forward(c_trg_sample)*self.batch_Ia[:, 3:]
                fake_image = torch.cat(
                    [fake_image, self.batch_Ia[:, 3:]], dim=1)
                # write_labels_on_images(fake_image,c_trg_sample)
                x_fake_list.append(fake_image)

            all_images.append(x_fake_list)

        x_concat = torch.cat(x_fake_list, dim=3)
        image = tvutils.make_grid(denorm(x_concat, device=self.device), nrow=1)
        tvutils.save_image(image, path)

    def testing_step(self, batch, batch_idx):
        self.eval_mode()
        i = batch_idx
        self.compute_sample_grid(batch,
                                 1.0,
                                 os.path.join(self.config.result_dir, "sample_{}_{}.png".format(
                                     i + 1, self.config.checkpoint)),
                                 writer=False)

    def output_results(self, batch, batch_id):

        self.batch_Ia, self.batch_normals, filename, self.batch_a_att = batch
        all_sample_list = self.create_interpolated_attr(
            self.batch_a_att, self.config.attrs, max_val=1)
        path = os.path.join(self.config.result_dir, str(
            self.config.checkpoint), "test")
        os.makedirs(path, exist_ok=True)
        batch_size = 32

        all_images = []
        for c_sample_list in all_sample_list:  # for each attr
            for c_trg_sample in c_sample_list:
                # print(c_trg_sample)
                fake_image = self.forward(c_trg_sample)*self.batch_Ia[:, 3:]
                fake_image = denorm(fake_image, device=self.device)
                fake_image = torch.cat(
                    [fake_image, self.batch_Ia[:, 3:]], dim=1)
                # print(fake_image.shape)
                # write_labels_on_images(fake_image,c_trg_sample)
                for i in range(fake_image.shape[0]):
                    attr_string = str(c_trg_sample[i].item()).replace('.', '_')

                    tvutils.save_image(fake_image[i],
                                       os.path.join(path, "{}_{}_{}.png".format(filename[i], self.config.attrs[0], attr_string)))

    # ================================================================================= #
    #                                 LOSSES UTILITIES                                  #
    # ================================================================================= #

    def regression_loss(self, logit, target):  # static
        return F.l1_loss(logit, target) / logit.size(0)

    def classification_loss(self, logit, target):  # static
        return F.cross_entropy(logit, target)

    def image_reconstruction_loss(self, Ia, Ia_hat):
        if self.config.rec_loss == 'l1':
            g_loss_rec = F.l1_loss(Ia, Ia_hat)
        elif self.config.rec_loss == 'l2':
            g_loss_rec = F.mse_loss(Ia, Ia_hat)
        elif self.config.rec_loss == 'perceptual':
            l1_loss = F.l1_loss(Ia, Ia_hat)
            self.scalars['G/loss_rec_l1'].append(l1_loss.item())
            g_loss_rec = l1_loss
            # add perceptual loss
            f_img = self.vgg16_f(Ia)
            f_img_hat = self.vgg16_f(Ia_hat)
            if self.config.lambda_G_perc > 0:
                self.scalars['G/loss_rec_perc'].append(self.config.lambda_G_perc *
                                                       self.loss_P(f_img_hat, f_img))
                g_loss_rec += self.scalars['G/loss_rec_perc'][-1].item()
            if self.config.lambda_G_style > 0:
                self.scalars['G/loss_rec_style'] = self.config.lambda_G_style * \
                    self.loss_S(f_img_hat, f_img)
                g_loss_rec += self.scalars['G/loss_rec_style'][-1].item()
        return g_loss_rec

    def angular_reconstruction_loss(self, normals, normals_hat):

        mask = normals[:, 3:]
        # with the mask
        return torch.sum(((normals[:, :3]-normals_hat)*mask)**2.0) / torch.sum(mask)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""

        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def to_multi_GPU(self, model):
        if self.config.ngpu > 1:
            return MyDataParallel(model, device_ids=list(range(self.config.ngpu)))
        else:
            return model

    def get_fadernet_output(self, att):
        with torch.no_grad():
            if self.config.image_size == 128:

                encodings, z, _ = self.G_small.encode(self.batch_Ia)
                fn_output, fn_features = self.G_small.decode_with_features(
                    att, z, self.get_normals(), encodings)
                return fn_output, fn_features, encodings

            else:
                rescaled_im = resize_right.resize(
                    self.batch_Ia, scale_factors=0.5)
                rescaled_normals = resize_right.resize(
                    self.get_normals(), scale_factors=0.5)

                encodings, z, _ = self.G_small.encode(rescaled_im)
                fn_output, fn_features = self.G_small.decode_with_features(
                    att, z, rescaled_normals, encodings)
                fn_output = resize_right.resize(fn_output, scale_factors=2)
                return fn_output, fn_features, encodings
