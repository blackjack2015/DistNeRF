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
from models.nerf import NeRF
from models.ray_utils import get_rays_np
import horovod.torch as hvd


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
        self.H, self.W, self.focal, self.K, self.images, self.poses, self.bds, self.render_poses, self.i_train, self.i_test, self.i_val, self.hwf, self.poses, self.near, self.far = self._prepare_dataset()
        self.model, self.step, self.grad_vars, self.optimizer = self._create_nerf()

    def named_parameters(self):
        """return the (name, parameters) of models.
           for DistributedOptimizer()
           may have two models
        """
        model = self.model.net
        model_fine = self.model.net_fine
        if self.args.N_importance > 0:
            full_model = nn.ModuleList(
                [model, model_fine]
            )
        else:
            full_model = model
        return full_model.named_parameters()

    def _create_nerf(self):
        """Instantiate NeRF's MLP model.
        """
    
        model = NeRF(H=self.H, W=self.W, focal=self.focal, K=self.K,
                     multires=self.args.multires, multires_views=self.args.multires_views,
                     i_embed=self.args.i_embed,
                     depth=self.args.netdepth, hidden=self.args.netwidth,
                     skips=[4], use_viewdirs=self.args.use_viewdirs,
                     near=self.near, far=self.far,
                     ndc=self.ndc, lindisp=self.lindisp,
                     N_samples=self.args.N_samples, N_importance=self.args.N_importance,
                     perturb=False, raw_noise_std=0., white_bkgd=self.args.white_bkgd,
                     ).cuda()
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

        start = 0
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
    
            start = ckpt['global_step'] + 1
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    
            # Load model
            model.load_state_dict(ckpt['network_state_dict'])
            # if model_fine is not None:
            #     model_fine.load_state_dict(ckpt['network_fine_state_dict'])
    
        ##########################
    
        return model, start, grad_vars, optimizer
    
    def get_current_step(self):
        return self.step

    def _prepare_dataset(self):

        # Load data
        K = None
        images = None
        poses = None
        bds = None
        render_poses = None
        i_train = None
        i_test = None
        i_val = None
        hwf = None
        poses = None
        near = 0.
        far = 0.
        if self.args.dataset_type == 'llff':
            images, poses, bds, render_poses, i_test = load_llff_data(self.args.datadir, self.args.factor,
                                                                      recenter=True, bd_factor=.75,
                                                                      spherify=self.args.spherify)
            hwf = poses[0,:3,-1]
            poses = poses[:,:3,:4]
            print('Loaded llff', images.shape, render_poses.shape, hwf, self.args.datadir)
            if not isinstance(i_test, list):
                i_test = [i_test]
    
            if self.args.llffhold > 0:
                print('Auto LLFF holdout,', self.args.llffhold)
                i_test = np.arange(images.shape[0])[::self.args.llffhold]
    
            i_val = i_test
            i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])
    
            print('DEFINING BOUNDS')
            if self.args.no_ndc:
                near = np.ndarray.min(bds) * .9
                far = np.ndarray.max(bds) * 1.
            else:
                near = 0.
                far = 1.
            print('NEAR FAR', near, far)
    
        elif self.args.dataset_type == 'blender':
            images, poses, render_poses, hwf, i_split = load_blender_data(self.args.datadir, self.args.half_res, self.args.testskip)
            print('Loaded blender', images.shape, render_poses.shape, hwf, self.args.datadir)
            i_train, i_val, i_test = i_split
    
            near = 2.
            far = 6.
    
            if self.args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]
    
        elif self.args.dataset_type == 'LINEMOD':
            images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(self.args.datadir, self.args.half_res, self.args.testskip)
            print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
            print(f'[CHECK HERE] near: {near}, far: {far}.')
            i_train, i_val, i_test = i_split
    
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]
    
        elif self.args.dataset_type == 'deepvoxels':
    
            images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                     basedir=args.datadir,
                                                                     testskip=args.testskip)
    
            print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
            i_train, i_val, i_test = i_split
    
            hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
            near = hemi_R-1.
            far = hemi_R+1.
    
        else:
            print('Unknown dataset type', args.dataset_type, 'exiting')
    
        # Cast intrinsics to right types
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
    
        if K is None:
            K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])
    
        # NDC only good for LLFF-style forward facing data
        if self.args.dataset_type != 'llff' or self.args.no_ndc:
            print('Not ndc!')
            self.ndc = False
        self.lindisp = self.args.lindisp
    
        return H, W, focal, K, images, poses, bds, render_poses, i_train, i_test, i_val, hwf, poses, near, far

    def prepare_train_env(self):

        # TODO: confirm what it does
        if self.args.render_test:
            self.render_poses = np.array(poses[i_test])
    
        # Create log dir and copy the config file
        f = os.path.join(self.basedir, self.expname, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(self.args)):
                attr = getattr(self.args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if self.args.config is not None:
            f = os.path.join(self.basedir, self.expname, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(self.args.config, 'r').read())
    
        # # needed if the version of pytorch >= 1.12
        # optimizer.param_groups[0]['capturable'] = True
    
        # Move testing data to GPU
        self.render_poses = torch.Tensor(self.render_poses).cuda()

        # Prepare raybatch tensor if batching random rays
        self.N_rand = self.args.N_rand
        self.use_batching = not self.args.no_batching
        if self.use_batching:
            # For random ray batching
            print('get rays')
            rays = np.stack([get_rays_np(self.H, self.W, self.K, p) for p in self.poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
            print('done, concats')
            rays_rgb = np.concatenate([rays, self.images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
            rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
            rays_rgb = np.stack([rays_rgb[i] for i in self.i_train], 0) # train images only
            rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
            rays_rgb = rays_rgb.astype(np.float32)
            print('shuffle rays')
            np.random.seed(5)
            np.random.shuffle(rays_rgb)
            self.i_batch = 0
    
            print('done')
    
        # Move training data to GPU
        if self.use_batching:
            self.rays_rgb = torch.Tensor(rays_rgb).cuda()
        self.images = torch.Tensor(self.images).cuda()
        self.poses = torch.Tensor(self.poses).cuda()

        print('Begin')
        print('TRAIN views are', self.i_train)
        print('TEST views are', self.i_test)
        print('VAL views are', self.i_val)
    
    def train_one_step(self, step):

        # Summary writers
        # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
        time0 = time.time()
        # Sample random ray batch
        if self.use_batching:
            # Random over all images
            samples_pw = self.rays_rgb.shape[0] // 4
            start_idx = self.rank * samples_pw + self.i_batch
            end_idx = start_idx + self.N_rand
            batch = self.rays_rgb[start_idx:end_idx] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]
    
            self.i_batch += self.N_rand
            if self.i_batch >= self.rays_rgb.shape[0] // 4:
                print("Shuffle data after an epoch!")
                # rand_idx = torch.randperm(self.rays_rgb.shape[0])
                # self.rays_rgb = self.rays_rgb[rand_idx]
                self.i_batch = 0
    
        else:
            # Random from one image
            img_i = np.random.choice(self.i_train)
            target = self.images[img_i]
            # target = torch.Tensor(target).cuda()
            pose = self.poses[img_i, :3,:4]
    
            if self.N_rand is not None:
                rays_o, rays_d = get_rays(self.H, self.W, self.K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
    
                if step < self.args.precrop_iters:
                    dH = int(self.H//2 * self.args.precrop_frac)
                    dW = int(self.W//2 * self.args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(self.H//2 - dH, self.H//2 + dH - 1, 2*dH), 
                            torch.linspace(self.W//2 - dW, self.W//2 + dW - 1, 2*dW)
                        ), -1)
                    # if step == self.step:
                    #     print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {self.args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, self.H-1, self.H), torch.linspace(0, self.W-1, self.W)), -1)  # (H, W, 2)
    
                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[self.N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    
        #####  Core optimization loop  #####
        rgb, disp, acc, extras = self.model.forward(rays=batch_rays, chunk=self.args.chunk)
    
        self.optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)
    
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)
    
        loss.backward()
        self.optimizer.step()

        dt = time.time()-time0
        self.step += 1

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

        step = self.step
        testsavedir = os.path.join(self.basedir, self.expname, 'testset_{:06d}'.format(step))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', self.poses[self.i_test].shape)
        with torch.no_grad():
            rgbs, _ = self.model.render(torch.Tensor(self.poses[self.i_test]).cuda(), self.args.chunk)
            img_loss = img2mse(rgbs, self.images[self.i_test][:rgbs.shape[0]])
            psnr = mse2psnr(img_loss)

            # save the results as png
            if testsavedir is not None:
                for i, rgb in enumerate(rgbs):
                    rgb8 = to8b(rgb.cpu().numpy())
                    filename = os.path.join(testsavedir, '{:03d}.png'.format(i))
                    imageio.imwrite(filename, rgb8)
    
        print('Saved test set')
        return psnr.item()
        
    def log_test_video(self):

        step = self.step
        # Turn on testing mode
        with torch.no_grad():
            rgbs, disps = self.model.render(self.render_poses, self.args.chunk)
        print('Done, saving', rgbs.shape, disps.shape)
        moviebase = os.path.join(self.basedir, self.expname, '{}_spiral_{:06d}_'.format(self.expname, step))
        rgbs, disps = rgbs.cpu().numpy(), disps.cpu().numpy()
        imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
    
        # if args.use_viewdirs:
        #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
        #     with torch.no_grad():
        #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
        #     render_kwargs_test['c2w_staticcam'] = None
        #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

    def log_checkpoint(self):
        # Rest is logging
        step = self.step
        path = os.path.join(self.basedir, self.expname, '{:06d}.tar'.format(step))
        torch.save({
            'global_step': step,
            'network_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print('Saved checkpoints at', path)
    
    
if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

