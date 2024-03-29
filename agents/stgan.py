from numpy.core.function_base import linspace
from tqdm import tqdm, trange
from utils.misc import print_cuda_statistics
from models.stgan import *
from datasets import *
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
from torch.backends import cudnn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import datetime
import time
import logging
import os
from utils.im_util import denorm
cudnn.benchmark = True
import time
from pytorch_msssim import ssim
import moviepy.video.io.ImageSequenceClip
import traceback

class STGANAgent(object):

    def __init__(self, config):
        
        self.config = config
        self.logger = logging.getLogger("STGAN")
        self.logger.info("Creating STGAN architecture...")

        self.G = Generator(len(self.config.attrs), self.config.g_conv_dim, self.config.g_layers,
                           self.config.shortcut_layers, use_stu=self.config.use_stu, one_more_conv=self.config.one_more_conv,deconv= self.config.deconv)
        if self.config.use_d:
            self.D = Discriminator(self.config.image_size, len(
            self.config.attrs), self.config.d_conv_dim, self.config.d_fc_dim, self.config.d_layers)

        if self.config.mode == 'plot_metrics':
            G_total_params = sum(p.numel() for p in self.G.parameters() if p.requires_grad)
            D_total_params = sum(p.numel() for p in self.D.parameters() if p.requires_grad)
            self.logger.info('Number of parameters of G:', G_total_params)
            self.logger.info('Number of parameters of D:',D_total_params)
            self.logger.info('Total parameters of the framework (G+D):',G_total_params+D_total_params)

        self.data_loader = globals()['{}_loader'.format(self.config.dataset)](
            self.logger, self.config.data_root, self.config.train_file, self.config.val_file, self.config.test_folder, self.config.mode, self.config.attrs,
            self.config.crop_size, self.config.image_size, self.config.batch_size, self.config.data_augmentation, mask_input_bg=config.mask_input_bg)

        self.current_iteration = 0
        self.current_epoch = 0
        self.cuda = torch.cuda.is_available() & self.config.cuda

        assert len (self.config.gpus.split(',')) == self.config.ngpu

        if self.cuda:

            self.device = torch.device("cuda:"+self.config.gpus)
            self.logger.info("Operation will be on *****GPU-CUDA***** ")
            print_cuda_statistics()

        else:
            
            self.device = torch.device("cpu")
            self.logger.info("Operation will be on *****CPU***** ")

        if self.config.mode == 'train':

            self.writer = SummaryWriter(log_dir=self.config.summary_dir)

    def save_checkpoint(self):
        G_state = {
            'state_dict': self.G.state_dict(),
            'optimizer': self.optimizer_G.state_dict()
        }
        D_state = {
            'state_dict': self.D.state_dict(),
            'optimizer': self.optimizer_D.state_dict()
        }
            
        G_filename = 'G_{}.pth.tar'.format(self.current_iteration)
        torch.save(G_state, os.path.join(
            self.config.checkpoint_dir, G_filename))
        

        if self.config.use_d:
            D_state = {
                'state_dict': self.D.state_dict(),
                'optimizer': self.optimizer_D.state_dict()
            }
            D_filename = 'D_{}.pth.tar'.format(self.current_iteration)
            torch.save(D_state, os.path.join(
            self.config.checkpoint_dir, D_filename))

    def make_video(self):

        """Compilate edited frames and makes a video."""
        fps = 25
        imgs = os.listdir(self.config.video_dir)
        imgs.sort()
        
        image_files = [os.path.join(self.config.video_dir,img)
                       for img in imgs
                       if img.endswith(".png")]
                 
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(os.path.join(self.config.video_dir,'video.mp4'))
                                
    def load_checkpoint(self):

        if self.config.checkpoint is None:
            self.G.to(self.device)

            if self.config.use_d:
                self.D.to(self.device)

            return
        if self.config.mode == 'train':

            G_filename = 'G_{}.pth.tar'.format(self.config.checkpoint)

        else:

            G_filename = '{}.pth.tar'.format(self.config.checkpoint)
        
        G_checkpoint = torch.load(os.path.join(self.config.checkpoint_dir, G_filename), map_location=self.device)
        G_to_load = {k.replace('module.', ''): v for k,
                     v in G_checkpoint['state_dict'].items()}
        self.G.load_state_dict(G_to_load)
        self.G.to(self.device)
        
        if self.config.mode == 'train':

            if self.config.use_d:

                D_filename = 'D_{}.pth.tar'.format(self.config.checkpoint)
                D_checkpoint = torch.load(os.path.join(
                self.config.checkpoint_dir, D_filename), map_location=self.device)
                D_to_load = {k.replace('module.', ''): v for k,
                     v in D_checkpoint['state_dict'].items()}
                self.D.load_state_dict(D_to_load)
                self.D.to(self.device)
                self.optimizer_D.load_state_dict(D_checkpoint['optimizer'])

            self.current_epoch = int(self.config.checkpoint // self.data_loader.train_iterations)
            self.current_iteration = self.config.checkpoint
            self.optimizer_G.load_state_dict(G_checkpoint['optimizer'])
            self.optimizer_D.load_state_dict(D_checkpoint['optimizer'])
        
    def create_interpolated_attr(self, c_org, selected_attrs=None,att_min=0, att_max=1, num_samples=9):
        
        """Generate target domain labels for debugging and testing: linearly sample attribute. Contains a list for each attr"""
        all_lists = []
        for i in range(len(selected_attrs)):
            c_trg_list = []  
            alphas = linspace(att_min, att_max, num_samples)
            for alpha in alphas:
                c_trg = c_org.clone()
                c_trg[:, i] = torch.full_like(c_trg[:, i], alpha)
                c_trg_list.append(c_trg)
            all_lists.append(c_trg_list)

        return all_lists

    def img2mse(self, x, y):

        """Compute Mean Square Error."""
        return torch.mean((x - y) ** 2)

    def img2mae(self, x, y):

        """Compute Mean Absolute Error."""
        return self.l1_loss(x, y)

    def mse2psnr(self, mse):

        """Compute PSNR."""
        return -10. * torch.log(mse) / torch.log(torch.Tensor([10.]).to(self.device))
    
    def l1_loss(self, logit, target):

        """Compute norm 1 loss."""
        return F.l1_loss(logit,target)

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

        assert self.config.mode in ['train', 'edit_images','edit_video', 'plot_metrics']
        try:
            if self.config.mode == 'train':

                self.train()

            elif self.config.mode == 'edit_images':
                
                # When editing in the wild, we asume the original value
                # for the attribute is 0.5
                idw_atts = torch.ones(1,len(self.config.attrs)) * 0.5
                c_trg_all = self.create_interpolated_attr(
                                                        idw_atts,
                                                        self.config.attrs, 
                                                        self.config.att_min, 
                                                        self.config.att_max, 
                                                        self.config.num_samples
                                                        )
                # Edit images
                self.edit_images(self.config.sample_dir, c_trg_all, video=False)

            elif self.config.mode == 'edit_video':
                
                # When editing in the wild, we asume the original value 
                # for the attribute is 0.5
                # Only one editing per frame
                idw_atts = torch.ones(1,len(self.config.attrs)) * 0.5
                #print(self.config.video_dir)
                os.system('rm -rf %s/*' % self.config.video_dir)
                #self.config.add_bg = True
                c_trg_all = self.create_interpolated_attr(
                                                        idw_atts,
                                                        self.config.attrs, 
                                                        self.config.att_value_frame, 
                                                        self.config.att_value_frame, 
                                                        self.config.num_samples
                                                        )
                # Edit frames
                self.edit_images(self.config.video_dir, c_trg_all, video=True)
                self.make_video()

            elif self.config.mode == 'plot_metrics':

                self.plot_metrics()

        except KeyboardInterrupt:
            self.logger.info('You have entered CTRL+C.. Wait to finalize')
        except Exception as e:
            
            log_file = open(os.path.join(self.config.log_dir, 'exp_error.log'), 'w+')
            traceback.print_exc(file=log_file)
            print(str(e))
        finally:

            self.finalize()

    def train(self):
        
        self.optimizer_G = optim.Adam(self.G.parameters(), self.config.g_lr, [
            self.config.beta1, self.config.beta2])


        if self.config.use_d:
            self.optimizer_D = optim.Adam(self.D.parameters(), self.config.d_lr, [
                self.config.beta1, self.config.beta2])

        self.load_checkpoint()

        if self.cuda and self.config.ngpu > 1:
            self.G = nn.DataParallel(
                self.G, device_ids=list(range(self.config.ngpu)))
            
            if self.config.use_d:
                self.D = nn.DataParallel(
                self.D, device_ids=list(range(self.config.ngpu)))

        val_iter = iter(self.data_loader.val_loader)

        x_sample, c_org_sample = next(val_iter)
        ch_4_sample = x_sample[:, 3:]
        x_sample = x_sample[:, :3]

        x_sample = x_sample.to(self.device)
        ch_4_sample = ch_4_sample.to(self.device)
        all_sample_list = self.create_interpolated_attr(
            c_org_sample, self.config.attrs, self.config.att_min,self.config.att_max, self.config.num_samples)

        if self.config.checkpoint:
            init_iteration = int(self.config.checkpoint % self.data_loader.train_iterations)
        else:
            init_iteration = 0

        data_iter = iter(self.data_loader.train_loader)
        start_time = time.time()
        
        self.G.train()

        if self.config.use_d:
            self.D.train()

        for epoch in range(self.current_epoch, self.config.max_epochs):
            
            for batch_id in trange(init_iteration, self.data_loader.train_iterations, desc='Epoch {}'.format(epoch),leave=(epoch==self.config.max_epochs-1)):
                
                # =================================================================================== #
                #                             0. Preprocess input data                                #
                # =================================================================================== #

                # fetch real images and labels
                try:
                    x_real, label_org = next(data_iter)
                except:
                    data_iter = iter(self.data_loader.train_loader)
                    x_real, label_org = next(data_iter)

                x_real = x_real.to(self.device)         # input images
                c_org = label_org.clone()

                x_real = x_real[:, :3]
                c_org = c_org.to(self.device)           # original domain labels
                # labels for computing classification loss
                label_org = label_org.to(self.device)
                scalars = {}


                # =================================================================================== #
                #                             1. Train the discriminator                              #
                # =================================================================================== #

                if self.config.use_d:

                    for i in range(self.config.n_critic):
                        # compute loss with real images
                        out_src, out_cls = self.D(x_real)
                        d_loss_real = - torch.mean(out_src)
                        d_loss_cls = self.l1_loss(out_cls, label_org) 

                        # generate target domain labels randomly or from the original data
                        if self.config.uniform:

                            label_trg = torch.rand_like(label_org)
                        else:

                            rand_idx = torch.randperm(label_org.size(0))
                            label_trg = label_org[rand_idx] 

                        # labels for computing classification loss
                        label_trg = label_trg.to(self.device)
                        c_trg = label_trg.clone()
                        c_trg = c_trg.to(self.device)           # target domain labels

                        # compute loss with fake images
                        x_fake = self.G(x_real, c_trg)
                            
                        out_src, out_cls = self.D(x_fake.detach())
                        d_loss_fake = torch.mean(out_src)

                        # compute loss for gradient penalty
                        alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                        x_hat = (alpha * x_real.data + (1 - alpha)* x_fake.data).requires_grad_(True)
                        out_src, _ = self.D(x_hat)
                        d_loss_gp = self.gradient_penalty(out_src, x_hat)

                        # backward and optimize
                        d_loss_adv = d_loss_real + d_loss_fake + self.config.lambda_gp * d_loss_gp
                        d_loss = d_loss_adv + self.config.lambda_1 * d_loss_cls
                        self.optimizer_D.zero_grad()
                        d_loss.backward(retain_graph=True)
                        self.optimizer_D.step()

                        # summarize
                        scalars['D/loss'] = d_loss.item()
                        scalars['D/loss_adv'] = d_loss_adv.item()
                        scalars['D/loss_cls'] = d_loss_cls.item()
                        scalars['D/loss_real'] = d_loss_real.item()
                        scalars['D/loss_fake'] = d_loss_fake.item()
                        scalars['D/loss_gp'] = d_loss_gp.item()
                
                # =================================================================================== #
                #                               2. Train the generator                                #
                # =================================================================================== #

                if self.config.uniform:

                    label_trg = torch.rand_like(label_org)
                else:

                    rand_idx = torch.randperm(label_org.size(0))
                    label_trg = label_org[rand_idx] 

                # labels for computing classification loss
                label_trg = label_trg.to(self.device)
                c_trg = label_trg.clone()
                c_trg = c_trg.to(self.device)           # target domain labels  
                
                # compute loss with fake images
                x_fake = self.G(x_real, c_trg)  

                if self.config.use_d:
                    out_src, out_cls = self.D(x_fake)
                    g_loss_adv = - torch.mean(out_src)
                    g_loss_cls = self.l1_loss(out_cls, label_trg) 

                # target-to-original domain
                x_reconst = self.G(x_real, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))
                
                # compute the PSNR
                img_mse = self.img2mse(x_reconst, x_real)
                psnr = self.mse2psnr(img_mse)

                # backward and optimize
                g_loss =   self.config.lambda_3 * g_loss_rec 
                
                if self.config.use_d:
                    g_loss += g_loss_adv + self.config.lambda_2 * g_loss_cls

                self.optimizer_G.zero_grad()
                g_loss.backward()
                self.optimizer_G.step()

                # summarize
                scalars['G/loss'] = g_loss.item()
                scalars['G/loss_rec'] = g_loss_rec.item()

                if self.config.use_d:
                    
                    scalars['G/loss_adv'] = g_loss_adv.item()
                    scalars['G/loss_cls'] = g_loss_cls.item()
                    scalars['G/psnr'] = psnr

                self.current_iteration += 1

                # =================================================================================== #
                #                                 3. Miscellaneous                                    #
                # =================================================================================== #

                if self.current_iteration % self.config.summary_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    print('Elapsed [{}], Iteration {}'.format(et,
                        self.current_iteration))
                    for tag, value in scalars.items():
                        self.writer.add_scalar(tag, value, self.current_iteration)

                if  self.current_iteration % self.config.sample_step == 0:
                        
                    self.G.eval()

                    with torch.no_grad():

                        att_idx = 0
                        for c_sample_list in all_sample_list:
                            
                            x_fake_list = [torch.cat([x_sample, ch_4_sample], dim=1)]

                            for c_trg_sample in c_sample_list:           
                                
                                x_fake = self.G(x_sample, c_trg_sample.to(self.device))
                                x_fake = torch.cat(
                                    [x_fake, ch_4_sample], dim=1)
                                x_fake_list.append(x_fake)

                            x_concat = torch.cat(x_fake_list, dim=3)
                            image = make_grid(denorm(x_concat, self.device,self.config.add_bg), nrow=1)
                            result_path = os.path.join(self.config.result_dir, 'sample_{}_{}.png'.format(self.config.attrs[att_idx],self.current_iteration))
                            save_image(image, result_path, nrow=1, padding=0)
                            del x_concat
                            del image
                            att_idx += 1
                            
                    self.G.train()
                    

                if  self.current_iteration % self.config.checkpoint_step == 0:
                    self.save_checkpoint()

                with torch.cuda.device('cuda:'+self.config.gpus):
                    torch.cuda.empty_cache()
            
            init_iteration = 0

    def add_background(self, x_edited, alpha_channel, mask, rgb):
        
        denorm_x_edited = denorm(x_edited, self.device, self.config.add_bg)[0]
        inv_mask = torch.where(mask.to(self.device) == 1, 0,1)
        denorm_x_edited = denorm_x_edited * inv_mask

        # Mask object and background
        alpha_channel_binary = torch.where(alpha_channel <= 0.6,  torch.tensor(1, dtype=alpha_channel.dtype).to(self.device) ,torch.tensor(0, dtype=alpha_channel.dtype).to(self.device))
        alpha_channel_binary_inv = torch.where(alpha_channel_binary == 0.,  torch.tensor(1, dtype=alpha_channel.dtype).to(self.device) ,torch.tensor(0, dtype=alpha_channel.dtype).to(self.device))
        background = rgb * alpha_channel_binary
        edited_object = denorm_x_edited * alpha_channel_binary_inv

        # Add background to the edited image
        image = edited_object + background
        image = torch.cat([image, torch.ones(1,rgb.shape[1],
                                         rgb.shape[2],device=self.device)], dim=0)
        return image

    def compute_metrics(self, real_image, rec_image):

        mse =self.img2mse(real_image,rec_image)
        mae = self.img2mae(real_image,rec_image)
        real_image = (real_image + 1) / 2 
        rec_image = (rec_image + 1) / 2 
        real_image = real_image.expand(1,4,256,256)
        rec_image = rec_image.expand(1,4,256,256)
        ssim_val = ssim( real_image, rec_image, data_range=1, size_average=False)

        return mse, mae, ssim_val
    
    def prepare_inference(self, msg_format):
        
        self.load_checkpoint()
        self.G.to(self.device)
        tqdm_loader = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations,
                           desc=msg_format)
        self.G.eval()

        return tqdm_loader
    
    def plot_metrics(self):

        psnr_file = open(os.path.join(self.config.metric_dir, 'psnr.txt'),'w')
        ssim_file = open(os.path.join(self.config.metric_dir, 'ssim.txt'),'w')
        mse_file = open(os.path.join(self.config.metric_dir, 'mse.txt'),'w')
        mae_file = open(os.path.join(self.config.metric_dir, 'mae.txt'),'w')

        mse_file.write('filename, MSE\n')
        mae_file.write('filename, MAE\n')     
        ssim_file.write('filename, SSIM\n')  
        psnr_file.write('filename, PSNR\n')

        tqdm_loader = self.prepare_inference('[METRICS]: Computing at checkpoint {}'.format(self.config.checkpoint))

        with torch.no_grad():

            for n_batch, (x_real, c_org, filename) in enumerate(tqdm_loader):
                
                att_idx = 0
                x_real = x_real.to(self.device)
                ch_4 = x_real[:, 3:]
                x_real = x_real[:, :3]
                x_rec = self.G(x_real, c_org.to(self.device))

                for n in range(0, x_real.shape[0]):                     
                                
                    rec_image = torch.cat([x_rec[n] , ch_4[n]], dim=0)
                    real_image = torch.cat([x_real[n] , ch_4[n]], dim=0)
                    mse, mae, ssim_val = self.compute_metrics(real_image.clone(), rec_image.clone())
                    
                    mse_file.write(filename[n] + ' ' + str(mse.item())+'\n')
                    mae_file.write(filename[n] + ' ' + str(mae.item())+'\n')     
                    ssim_file.write(filename[n] + ' ' + str(ssim_val.item() )+'\n')  
                    psnr_file.write(filename[n] + ' ' + str(self.mse2psnr(mse).item())+'\n')

                    real_image = denorm(real_image, self.device,self.config.add_bg)[0]
                    rec_image = denorm(rec_image, self.device,self.config.add_bg)[0]

                    name = 'Image_{}_{}.png'.format(n + x_real.shape[0] * n_batch,self.config.attrs[att_idx])
                    result_path = os.path.join(self.config.metric_dir, name)
                    image_pair = torch.cat([real_image, rec_image], dim=2)
                    save_image(image_pair, result_path, nrow=1, padding=0)

     
        psnr_file.close()   
        ssim_file.close()
        mse_file.close()
        mae_file.close()

    def edit_images(self, output_path, c_trg_all, video=False):

        editing_type = 'VIDEO' if video else 'IMAGE'
        tqdm_loader = self.prepare_inference('[{} EDITING]: Attribute {}'.format(editing_type, self.config.checkpoint))
        with torch.no_grad():

            for n_batch, (x_real, mask,rgb, filename) in enumerate(tqdm_loader):

                rgb = rgb.to(self.device) 
                att_idx = 0
                x_real = x_real.to(self.device)
                ch_4 = x_real[:, 3:]
                x_edited_list = []

                for c_trg_list in c_trg_all:

                    # Save original image
                    if self.config.add_bg:

                        x_edited_list.append([torch.cat([rgb[0], 
                                                       torch.ones((1,rgb[0].shape[1],
                                                       rgb[0].shape[2]),
                                                       device=self.device)], dim=0)])
                    else:

                        x_real_denorm = denorm(x_real, self.device, False)
                        x_real_denorm = x_real_denorm[:,:3] 
                        x_edited_list.append([torch.cat([x_real_denorm[0], ch_4[0]], dim=0)])

                    for c_trg_sample in c_trg_list:

                        x_real = x_real[:, :3]
                        x_edited = self.G(x_real, c_trg_sample.to(self.device))

                        for n in range(0, x_real.shape[0]):                     
                                
                            if self.config.add_bg:

                                final_image = self.add_background(x_edited[n], ch_4[n], mask, rgb[n])

                            else:
                                    
                                final_image = torch.cat([x_edited[n] , ch_4[n]], dim=0)
                                final_image = denorm(final_image, self.device,self.config.add_bg)[0]
                                
                            x_edited_list[n].append(final_image)
                            x_concat = torch.cat(x_edited_list[n], dim=2)

                    # Save edited images
                    if not video:

                        name = '{}.png'.format(filename[0])
                    else:
                        name = 'Frame_{}_{:05d}.png'.format(self.config.attrs[att_idx], n + x_real.shape[0] * n_batch)
                    result_path = os.path.join(output_path, name)
                    edited_images = make_grid(x_concat, nrow=1)
                    save_image(edited_images, result_path, nrow=1, padding=0)
                    att_idx += 1


    def finalize(self):

        self.logger.info('Please wait while finalizing the operation... Thank you :D')

        if self.config.mode == 'train':

            self.writer.export_scalars_to_json(os.path.join(self.config.summary_dir, 'all_scalars.json'))
            self.writer.close()
