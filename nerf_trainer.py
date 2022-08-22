import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib.pyplot as plt

from lamb import Lamb

from dataloader.load_llff import load_llff_data
from dataloader.load_deepvoxels import load_dv_data
from dataloader.load_blender import load_blender_data
from dataloader.load_LINEMOD import load_LINEMOD_data
from models import get_model
from models.ray_utils import get_rays_np
from models.AverageMeter import AverageMeter, HVDMetric
import horovod.torch as hvd
from dataloader.datasets import cycle, get_dataloader, ray_to_device


# Misc
DEBUG = False
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).cuda())
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


class NerfTrainer:

    def __init__(self, args, rank, local_rank, nworkers=1, net_arch='nerf'):

        # distributed info
        self.rank = rank
        self.local_rank = local_rank
        self.nworkers = nworkers

        # TODO: argument input re-organization
        self.args = args

        # NeRF info
        self.net_arch = net_arch

        # initialization
        # self.H, self.W, self.focal, self.K, self.images, self.poses, self.bds, self.render_poses, self.i_train, self.i_test, self.i_val, self.hwf, self.poses, self.near, self.far = self._prepare_dataset()
        self.train_loader, self.eval_loader = self._prepare_dataloader()
        self.model, self.epoch, self.grad_vars, self.optimizer = self._create_nerf()

    def named_parameters(self):
        """return the (name, parameters) of models.
           for DistributedOptimizer()
           may have two models
        """
        # model = self.model.net
        # model_fine = self.model.net_fine
        # if self.args.N_importance > 0:
        #     full_model = nn.ModuleList(
        #         [model, model_fine]
        #     )
        # else:
        #     full_model = model
        # return full_model.named_parameters()
        return self.model.named_parameters()

    def _create_nerf(self):
        """Instantiate NeRF's MLP model.
        """
    
        model = get_model(self.net_arch, self.args)

        grad_vars = list(model.parameters())
    
        # Create optimizer
        lrate = self.args.lrate
        if self.nworkers == 1:
            optimizer = torch.optim.Adam(params=grad_vars, lr=lrate, betas=(0.9, 0.999))
        else:
            lrate *= self.nworkers
            optimizer = torch.optim.Adam(params=grad_vars, lr=lrate, betas=(0.9, 0.999))
            # optimizer = Lamb(params=grad_vars, lr=lrate, betas=(0.9, 0.999))
        print("initial lr:", lrate)

        epoch = 0
        self.basedir = self.args.basedir
        self.expname = self.args.expname
        os.makedirs(os.path.join(self.basedir, self.expname), exist_ok=True)

        print(self.basedir, self.expname)
    
        ##########################
    
        # Load checkpoints
        if self.args.ft_path is not None and self.args.ft_path!='None':
            ckpts = [self.args.ft_path]
        else:
            ckpts = [os.path.join(self.basedir, self.expname, f) for f in sorted(os.listdir(os.path.join(self.basedir, self.expname))) if 'tar' in f]
    
        print('Found ckpts', ckpts)
        if len(ckpts) > 0 and not self.args.no_reload:
            ckpt_path = ckpts[-1]
            print('Reloading from', ckpt_path)
            ckpt = torch.load(ckpt_path)
    
            epoch = ckpt['epoch'] + 1
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    
            # Load model
            model.load_state_dict(ckpt['network_state_dict'])
            # if model_fine is not None:
            #     model_fine.load_state_dict(ckpt['network_fine_state_dict'])
    
        ##########################
    
        return model, epoch, grad_vars, optimizer
    
    def get_current_epoch(self):
        return self.epoch

    def _prepare_dataloader(self):

        train_loader = get_dataloader(dataset_name=self.args.dataset_type, base_dir=self.args.datadir, split="train", factor=2., batch_size=self.args.N_rand, shuffle=True, device=torch.device("cuda"), nworkers=self.nworkers, rank=self.rank)
        eval_loader = get_dataloader(dataset_name=self.args.dataset_type, base_dir=self.args.datadir, split="test", factor=2., batch_size=self.args.N_rand, shuffle=False, device=torch.device("cuda"), nworkers=self.nworkers, rank=self.rank)

        return train_loader, eval_loader

    def load_next_batch(self):

        return next(self.train_loader)

    def train_one_epoch(self, epoch):

        self.epoch = epoch
        self.model.train()

        losses = AverageMeter()
        psnrs = AverageMeter()
        if self.rank == 0:
            iterator = tqdm(self.train_loader)
        else:
            iterator = self.train_loader
        for i, batch in enumerate(iterator):

            loss, psnr = self.train_one_step(batch)
            losses.update(loss)
            psnrs.update(psnr)

            if self.rank == 0:
                iterator.set_postfix(loss=loss, psnr=psnr, avg_psnr=psnrs.avg)

        return losses.avg, psnrs.avg

    def train_one_step(self, batch):

        time0 = time.time()

        rays, target_s = batch
        rays = ray_to_device(rays)
        target_s = target_s.cuda()
        #####  Core optimization loop  #####
        rgb, disp, acc, extras = self.model(rays=rays, chunk=self.args.chunk)

        self.optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        # trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)
    
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)
        if 'raw' in extras:
            trans = extras['raw'][...,-1]
    
        loss.backward()
        self.optimizer.step()

        dt = time.time()-time0

        return loss.item(), psnr.item()

    def update_learning_rate(self, global_step):

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = self.args.lrate_decay * 1000
        new_lrate = self.args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################
    
            # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
            #####           end            #####
    
    def log_test_set(self):

        epoch = self.epoch
        testsavedir = os.path.join(self.basedir, self.expname, 'testset_{:02d}'.format(epoch))
        os.makedirs(testsavedir, exist_ok=True)
        # print('test poses shape', self.poses[self.i_test].shape)
        rgbs = []
        disps = []
        if self.nworkers > 1:
            losses = HVDMetric('val_loss')
            psnrs = HVDMetric('val_psnr')
        else:
            losses = AverageMeter()
            psnrs = AverageMeter()

        self.model.eval()
        if self.rank == 0:
            iterator = tqdm(self.eval_loader)
        else:
            iterator = self.eval_loader
        for i, batch in enumerate(iterator):
            rays, pixels = batch
            rays = ray_to_device(rays)
            pixels = pixels.cuda()  # [height*width, 3]

            with torch.no_grad():
                rgb, disp, acc = self.model.render(rays, self.eval_loader.h, self.eval_loader.w)  # rgbs: [height, width, 3]
            rgbs.append(rgb)
            disps.append(disp)

            rgb = rgb.reshape(pixels.size()[0], 3)
            disp = disp.reshape(pixels.size()[0], 1)

            img_loss = img2mse(rgb, pixels)
            psnr = mse2psnr(img_loss)

            losses.update(img_loss)
            psnrs.update(psnr)

            if self.rank == 0:
                iterator.set_postfix(loss=img_loss.item(), psnr=psnr.item(), avg_psnr=psnrs.avg)

            # # re-arrange the pixels
            # indices = np.arange(pixels.size()[0] * pixels.size()[1])
            # pixels_per_worker = (pixels.size()[0] * pixels.size()[1] + 1) // self.nworkers
            # indices = indices // pixels_per_worker + indices % pixels_per_worker * self.nworkers
            # rgb = rgb[indices]
            # disp = disp[indices]
            # rgb = rgb.reshape(self.eval_loader.h, self.eval_loader.w, 3)
            # disp = disp.reshape(self.eval_loader.h, self.eval_loader.w, 3)

        # save the results as png
        if testsavedir is not None:
            for i, rgb in enumerate(rgbs):
                rgb8 = to8b(rgb.cpu().numpy())
                filename = os.path.join(testsavedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)
    
        # save test images to a video
        moviebase = os.path.join(self.basedir, self.expname, '{}_spiral_{:02d}_'.format(self.expname, self.epoch))
        rgbs, disps = torch.stack(rgbs, 0).cpu().numpy(), torch.stack(disps, 0).cpu().numpy()
        imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
    
        return psnrs.avg
        
    def log_checkpoint(self):
        # Rest is logging
        path = os.path.join(self.basedir, self.expname, '{:02d}.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'network_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print('Saved checkpoints at', path)
    
    
if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

