from utils.im_util import denorm
import os
import logging
import time
import datetime
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from tensorboardX import SummaryWriter

from datasets import *
from models.stgan import Generator, Discriminator
from utils.misc import print_cuda_statistics
from tqdm import tqdm, trange
from numpy.core.function_base import linspace
cudnn.benchmark = True

att_diff = True


class STGANAgent(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("STGAN")
        self.logger.info("Creating STGAN architecture...")

        self.G = Generator(len(self.config.attrs), self.config.g_conv_dim, self.config.g_layers,
                           self.config.shortcut_layers, use_stu=self.config.use_stu, one_more_conv=self.config.one_more_conv)
        self.D = Discriminator(self.config.image_size, len(
            self.config.attrs), self.config.d_conv_dim, self.config.d_fc_dim, self.config.d_layers)

        self.data_loader = globals()['{}_loader'.format(self.config.dataset)](
            self.config.data_root, self.config.train_file, self.config.test_file, self.config.mode, self.config.attrs,
            self.config.crop_size, self.config.image_size, self.config.batch_size, self.config.data_augmentation, mask_input_bg=config.mask_input_bg)

        self.current_iteration = 0
        self.cuda = torch.cuda.is_available() & self.config.cuda

        if self.cuda:
            self.device = torch.device("cuda:1")
            self.logger.info("Operation will be on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            self.logger.info("Operation will be on *****CPU***** ")

        self.writer = SummaryWriter(log_dir=self.config.summary_dir)

    def save_checkpoint(self):
        G_state = {
            'state_dict': self.G.state_dict(),
            'optimizer': self.optimizer_G.state_dict(),
        }
        D_state = {
            'state_dict': self.D.state_dict(),
            'optimizer': self.optimizer_D.state_dict(),
        }
        G_filename = 'G_{}.pth.tar'.format(self.current_iteration)
        D_filename = 'D_{}.pth.tar'.format(self.current_iteration)
        torch.save(G_state, os.path.join(
            self.config.checkpoint_dir, G_filename))
        torch.save(D_state, os.path.join(
            self.config.checkpoint_dir, D_filename))

    def load_checkpoint(self):
        if self.config.checkpoint is None:
            self.G.to(self.device)
            self.D.to(self.device)
            return
        G_filename = 'G_{}.pth.tar'.format(self.config.checkpoint)
        D_filename = 'D_{}.pth.tar'.format(self.config.checkpoint)
        G_checkpoint = torch.load(os.path.join(
            self.config.checkpoint_dir, G_filename))
        D_checkpoint = torch.load(os.path.join(
            self.config.checkpoint_dir, D_filename))
        G_to_load = {k.replace('module.', ''): v for k,
                     v in G_checkpoint['state_dict'].items()}
        D_to_load = {k.replace('module.', ''): v for k,
                     v in D_checkpoint['state_dict'].items()}
        self.current_iteration = self.config.checkpoint
        self.G.load_state_dict(G_to_load)
        self.D.load_state_dict(D_to_load)
        self.G.to(self.device)
        self.D.to(self.device)
        if self.config.mode == 'train':
            self.optimizer_G.load_state_dict(G_checkpoint['optimizer'])
            self.optimizer_D.load_state_dict(D_checkpoint['optimizer'])

    def create_interpolated_attr(self, c_org, selected_attrs=None, max_val=5.0):
        """Generate target domain labels for debugging and testing: linearly sample attribute. Contains a list for each attr"""
        all_lists = []
        for i in range(len(selected_attrs)):
            c_trg_list = []  # [c_org]
            alphas = [-max_val, -((max_val-1)/2.0+1), -1, -
                      0.5, 0, 0.5, 1, ((max_val-1)/2.0+1), max_val]
            if max_val == 1:
                alphas = linspace(-1, 2, 9)  # [-1,-0.75,-0.5,0,0.5,1,]
            # alphas = np.linspace(-max_val, max_val, 10)
            for alpha in alphas:
                c_trg = c_org.clone()
                c_trg[:, i] = torch.full_like(c_trg[:, i], alpha)
                c_trg_list.append(c_trg)
            all_lists.append(c_trg_list)

        return all_lists

    def img2mse(self, x, y):
        return torch.mean((x - y) ** 2)

    def mse2psnr(self, x):
        return -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(self.device))

    def classification_loss(self, logit, target):
        """Compute binary cross entropy loss."""
        return F.binary_cross_entropy_with_logits(logit, target, reduction='sum') / logit.size(0)

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

    def run(self):
        assert self.config.mode in ['train', 'test']
        # try:
        if self.config.mode == 'train':

            self.train()
        else:
            self.test()
        # except KeyboardInterrupt:
            # self.logger.info('You have entered CTRL+C.. Wait to finalize')
        # except Exception as e:
            # log_file = open(os.path.join(
            #    self.config.log_dir, 'exp_error.log'), 'w+')
            # traceback.print_exc(file=log_file)
            # print(str(e))
        # finally:
            # self.finalize()

    def train(self):
        self.optimizer_G = optim.Adam(self.G.parameters(), self.config.g_lr, [
            self.config.beta1, self.config.beta2])
        self.optimizer_D = optim.Adam(self.D.parameters(), self.config.d_lr, [
            self.config.beta1, self.config.beta2])
        self.lr_scheduler_G = optim.lr_scheduler.StepLR(
            self.optimizer_G, step_size=self.config.lr_decay_iters, gamma=0.1)
        self.lr_scheduler_D = optim.lr_scheduler.StepLR(
            self.optimizer_D, step_size=self.config.lr_decay_iters, gamma=0.1)

        self.load_checkpoint()

        if self.cuda and self.config.ngpu > 1:
            self.G = nn.DataParallel(
                self.G, device_ids=list(range(self.config.ngpu)))
            self.D = nn.DataParallel(
                self.D, device_ids=list(range(self.config.ngpu)))

        val_iter = iter(self.data_loader.val_loader)

        x_sample, c_org_sample = next(val_iter)
        ch_4_sample = x_sample[:, 3:]
        x_sample = x_sample[:, :3]

        x_sample = x_sample.to(self.device)
        ch_4_sample = ch_4_sample.to(self.device)
        all_sample_list = self.create_interpolated_attr(
            c_org_sample, self.config.attrs, max_val=1.0)

        # c_sample_list.insert(0, c_org_sample)  # reconstruction

        self.g_lr = self.lr_scheduler_G.get_lr()[0]
        self.d_lr = self.lr_scheduler_D.get_lr()[0]

        data_iter = iter(self.data_loader.train_loader)
        start_time = time.time()
        for i in trange(self.current_iteration, self.config.max_iters):
            self.G.train()
            self.D.train()

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # fetch real images and labels
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(self.data_loader.train_loader)
                x_real, label_org = next(data_iter)
            x_real = x_real[:, :3]

            # generate target domain labels randomly
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            c_org = label_org.clone()
            c_trg = label_trg.clone()

            x_real = x_real.to(self.device)         # input images
            c_org = c_org.to(self.device)           # original domain labels
            c_trg = c_trg.to(self.device)           # target domain labels
            # labels for computing classification loss
            label_org = label_org.to(self.device)
            # labels for computing classification loss
            label_trg = label_trg.to(self.device)

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # compute loss with real images
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org)

            # compute loss with fake images
            if att_diff:
                attr_diff = c_trg - c_org
                attr_diff = attr_diff * \
                    torch.rand_like(attr_diff) * (2 * self.config.thres_int)
                x_fake = self.G(x_real, attr_diff)

            else:
                attr_fake = torch.rand_like(c_org)
                x_fake = self.G(x_real, attr_fake)

            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # compute loss for gradient penalty
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)

            x_hat = (alpha * x_real.data + (1 - alpha)
                     * x_fake.data).requires_grad_(True)

            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # backward and optimize
            d_loss_adv = d_loss_real + d_loss_fake + self.config.lambda_gp * d_loss_gp
            d_loss = d_loss_adv + self.config.lambda1 * d_loss_cls
            self.optimizer_D.zero_grad()
            d_loss.backward(retain_graph=True)
            self.optimizer_D.step()

            # summarize
            scalars = {}
            scalars['D/loss'] = d_loss.item()
            scalars['D/loss_adv'] = d_loss_adv.item()
            scalars['D/loss_cls'] = d_loss_cls.item()
            scalars['D/loss_real'] = d_loss_real.item()
            scalars['D/loss_fake'] = d_loss_fake.item()
            scalars['D/loss_gp'] = d_loss_gp.item()

            del d_loss
            del out_src
            del out_cls

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.config.n_critic == 0:
                # original-to-target domain
                if att_diff:
                    x_fake = self.G(x_real, attr_diff)
                else:
                    attr_fake = torch.rand_like(c_org)
                    x_fake = self.G(x_real, attr_fake)

                out_src, out_cls = self.D(x_fake)
                g_loss_adv = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg)

                # target-to-original domain
                if att_diff:
                    x_reconst = self.G(x_real, c_org - c_org)
                else:
                    x_reconst = self.G(x_real, c_org)

                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # compute the PSNR
                img_mse = self.img2mse(x_reconst, x_real)
                psnr = self.mse2psnr(img_mse)

                # backward and optimize
                g_loss = g_loss_adv + self.config.lambda3 * \
                    g_loss_rec + self.config.lambda2 * g_loss_cls
                self.optimizer_G.zero_grad()
                g_loss.backward()
                self.optimizer_G.step()

                # summarize
                scalars['G/loss'] = g_loss.item()
                scalars['G/loss_adv'] = g_loss_adv.item()
                scalars['G/loss_cls'] = g_loss_cls.item()
                scalars['G/loss_rec'] = g_loss_rec.item()
                scalars['G/psnr'] = psnr

                del g_loss
                del x_fake
                del x_reconst

            self.current_iteration += 1

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            if self.current_iteration % self.config.summary_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                print('Elapsed [{}], Iteration [{}/{}]'.format(et,
                      self.current_iteration, self.config.max_iters))
                for tag, value in scalars.items():
                    self.writer.add_scalar(tag, value, self.current_iteration)

            if self.current_iteration % self.config.sample_step == 0:
                self.G.eval()
                with torch.no_grad():

                    for c_sample_list in all_sample_list:

                        x_fake_list = [torch.cat(
                            [x_sample, ch_4_sample], dim=1)]

                        for c_trg_sample in c_sample_list:

                            if att_diff:
                                attr_diff = (c_trg_sample -
                                             c_org_sample).to(self.device)
                                x_fake = self.G(
                                    x_sample, attr_diff.to(self.device))
                            else:
                                x_fake = self.G(
                                    x_sample, c_trg_sample.to(self.device))

                            x_fake = torch.cat(
                                [x_fake, ch_4_sample], dim=1)
                            x_fake_list.append(x_fake)

                        x_concat = torch.cat(x_fake_list, dim=3)
                        image = make_grid(
                            denorm(x_concat, device=self.device), nrow=1)
                        self.writer.add_image('sample', make_grid(
                            denorm(x_concat, device=self.device), nrow=1), self.current_iteration)

                        del x_concat
                        del image

            if self.current_iteration % self.config.checkpoint_step == 0:
                self.save_checkpoint()

            self.lr_scheduler_G.step()
            self.lr_scheduler_D.step()
            with torch.cuda.device('cuda:1'):
                torch.cuda.empty_cache()

    def test(self):
        self.load_checkpoint()
        self.G.to(self.device)

        tqdm_loader = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations,
                           desc='Testing at checkpoint {}'.format(self.config.checkpoint))

        self.G.eval()
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(tqdm_loader):
                x_real = x_real.to(self.device)
                ch_4 = x_real[:, 3:]
                x_real = x_real[:, :3]

                c_trg_all = self.create_interpolated_attr(
                    c_org, self.config.attrs, max_val=1.0)

                for c_trg_list in c_trg_all:

                    if self.config.split:
                        x_fake_list = []
                        for n in range(0, x_real.shape[0]):

                            x_fake_list.append(
                                [torch.cat([x_real[n], ch_4[n]], dim=0)])

                    else:
                        x_fake_list = [torch.cat([x_real, ch_4], dim=1)]

                    for c_trg_sample in c_trg_list:

                        if att_diff:
                            attr_diff = c_trg_sample.to(
                                self.device) - c_org.to(self.device)
                            x_fake = self.G(x_real, attr_diff.to(self.device))
                        else:
                            x_fake = self.G(
                                x_real, c_trg_sample.to(self.device))

                        if self.config.split:

                            for n in range(0, x_real.shape[0]):

                                x_fake_n = torch.cat(
                                    [x_fake[n], ch_4[n]], dim=0)
                                x_fake_list[n].append(x_fake_n)

                        else:

                            x_fake = torch.cat([x_fake, ch_4], dim=1)
                            x_fake_list.append(x_fake)

                    if self.config.split:
                        for n in range(0, x_real.shape[0]):

                            x_concat = torch.cat(x_fake_list[n], dim=2)
                            image = make_grid(
                                denorm(x_concat, device=self.device), nrow=1)
                            result_path = os.path.join(
                                self.config.result_dir, 'sample_{}_{}.png'.format(n, i + 1))
                            save_image(image, result_path, nrow=1, padding=0)
                    else:
                        x_concat = torch.cat(x_fake_list, dim=3)
                        image = make_grid(
                            denorm(x_concat, device=self.device), nrow=1)
                        result_path = os.path.join(
                            self.config.result_dir, 'sample_{}.png'.format(i + 1))
                        save_image(image, result_path, nrow=1, padding=0)

    def finalize(self):
        print('Please wait while finalizing the operation.. Thank you')
        self.writer.export_scalars_to_json(os.path.join(
            self.config.summary_dir, 'all_scalars.json'))
        self.writer.close()
