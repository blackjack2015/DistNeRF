import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from models.ray_utils import get_rays, ndc_rays, sample_pdf

DEBUG = False

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

# NLP Model
class BasicMLP(nn.Module):
    def __init__(self,
                 depth=8,
                 hidden=256,
                 input_ch=3,
                 input_ch_views=3,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False,
                 ):

        super(BasicMLP, self).__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.output_ch = output_ch
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, hidden)] + [nn.Linear(hidden, hidden) if i not in self.skips else nn.Linear(hidden + input_ch, hidden) for i in range(depth-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + hidden, hidden//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(hidden, hidden)
            self.alpha_linear = nn.Linear(hidden, 1)
            self.rgb_linear = nn.Linear(hidden//2, 3)
        else:
            self.output_linear = nn.Linear(hidden, output_ch)

    def forward(self, x):

        # network forward
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    


# NeRF Model
class NeRF(nn.Module):
    def __init__(self,
                 H,
                 W,
                 focal,
                 K,
                 multires,
                 multires_views,
                 i_embed,
                 depth=8,
                 hidden=256,
                 skips=[4],
                 use_viewdirs=False,
                 near=0.,
                 far=1.,
                 ndc=True,
                 lindisp=True,
                 N_samples=64,
                 N_importance=128,
                 perturb=0.,
                 raw_noise_std=0.,
                 white_bkgd=False,
                 ):
        """ 
        """
        super(NeRF, self).__init__()
        self.H = H
        self.W = W
        self.focal = focal
        self.K = K
        self.depth = depth
        self.hidden = hidden
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.near = near
        self.far = far
        self.ndc = ndc
        self.lindisp = lindisp
        self.N_samples = N_samples
        self.N_importance = N_importance
        self.white_bkgd = white_bkgd

        # can be different between train and test
        self.perturb = perturb
        self.raw_noise_std = raw_noise_std

        # positional encoding modules
        self.embed_fn, self.input_ch = get_embedder(multires, i_embed)
        self.input_ch_views = 0
        self.embeddirs_fn = None
        if use_viewdirs:
            self.embeddirs_fn, self.input_ch_views = get_embedder(multires_views, i_embed)
        self.output_ch = 5 if N_importance > 0 else 4

        # MLP modules
        
        self.net = BasicMLP(depth, hidden, self.input_ch, self.input_ch_views, self.output_ch, skips, use_viewdirs)
        self.net_fine = None
        if N_importance > 0:
            self.net_fine = BasicMLP(depth, hidden, self.input_ch, self.input_ch_views, self.output_ch, skips, use_viewdirs)
        
    def batchify(self, fn, chunk):
        """Constructs a version of 'fn' that applies to smaller batches.
        """
        if chunk is None:
            return fn
        def ret(inputs):
            return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
        return ret
    
    def batchify_rays(self, rays_flat, chunk=1024*32, retraw=False):
        """Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = {}
        for i in range(0, rays_flat.shape[0], chunk):
            ret = self.render_rays(rays_flat[i:i+chunk], retraw=retraw)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
    
        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret
    
    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    
        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).cuda()], -1)  # [N_rays, N_samples]
    
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    
        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[...,3].shape) * raw_noise_std
    
            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
                noise = torch.Tensor(noise)
            noise = noise.cuda()
    
        alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    
        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)
    
        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[...,None])
    
        return rgb_map, disp_map, acc_map, weights, depth_map
    
    def render_rays(self,
                    ray_batch,
                    retraw=False,
                    verbose=False,
                    pytest=False):
        """Volumetric rendering.
        Args:
          ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
          network_fn: function. Model for predicting RGB and density at each point
            in space.
          network_query_fn: function used for passing queries to network_fn.
          N_samples: int. Number of different times to sample along each ray.
          retraw: bool. If True, include model's raw, unprocessed predictions.
          lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
          perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
          N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
          network_fine: "fine" network with same spec as network_fn.
          white_bkgd: bool. If True, assume a white background.
          raw_noise_std: ...
          verbose: bool. If True, print more debugging info.
        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
          disp_map: [num_rays]. Disparity map. 1 / depth.
          acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
          raw: [num_rays, num_samples, 4]. Raw predictions from model.
          rgb0: See rgb_map. Output for coarse model.
          disp0: See disp_map. Output for coarse model.
          acc0: See acc_map. Output for coarse model.
          z_std: [num_rays]. Standard deviation of distances along ray for each
            sample.
        """
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
        viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
        near, far = bounds[...,0], bounds[...,1] # [-1,1]
    
        t_vals = torch.linspace(0., 1., steps=self.N_samples).cuda()
        if not self.lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    
        z_vals = z_vals.expand([N_rays, self.N_samples])
    
        if self.perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)
    
            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand)
    
            t_rand = t_rand.cuda()
            z_vals = lower + (upper - lower) * t_rand
    
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    
    
        raw = self.run_network(pts, viewdirs, self.net)
        rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(raw, z_vals, rays_d, self.raw_noise_std, self.white_bkgd, pytest=pytest)
    
        if self.N_importance > 0:
    
            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
    
            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], self.N_importance, det=(self.perturb==0.), pytest=pytest)
            z_samples = z_samples.detach()
    
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
    
            run_fn = self.net if self.net_fine is None else self.net_fine
            raw = self.run_network(pts, viewdirs, run_fn)
    
            rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(raw, z_vals, rays_d, self.raw_noise_std, self.white_bkgd, pytest=pytest)
    
        ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
        if retraw:
            ret['raw'] = raw
        if self.N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
    
        for k in ret:
            if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
                print(f"! [Numerical Error] {k} contains nan or inf.")
    
        return ret

    def run_network(self, inputs, viewdirs, net, netchunk=1024*64):
        """Prepares inputs and applies network 'fn'.
        """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = self.embed_fn(inputs_flat)
    
        if viewdirs is not None:
            input_dirs = viewdirs[:,None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = self.embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)
    
        outputs_flat = self.batchify(net, netchunk)(embedded)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs
    
    def forward(self, rays=None, c2w=None, c2w_staticcam=None, chunk=1024*32):
        """Render rays
        Args:
          H: int. Height of image in pixels.
          W: int. Width of image in pixels.
          focal: float. Focal length of pinhole camera.
          chunk: int. Maximum number of rays to process simultaneously. Used to
            control maximum memory usage. Does not affect final results.
          rays: array of shape [2, batch_size, 3]. Ray origin and direction for
            each example in batch.
          c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
          ndc: bool. If True, represent ray origin, direction in NDC coordinates.
          near: float or array of shape [batch_size]. Nearest distance for a ray.
          far: float or array of shape [batch_size]. Farthest distance for a ray.
          use_viewdirs: bool. If True, use viewing direction of a point in space in model.
          c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
           camera while using other c2w argument for viewing directions.
        Returns:
          rgb_map: [batch_size, 3]. Predicted RGB values for rays.
          disp_map: [batch_size]. Disparity map. Inverse of depth.
          acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
          extras: dict with everything returned by render_rays().
        """
        if c2w is not None:
            # special case to render full image
            rays_o, rays_d = get_rays(self.H, self.W, self.K, c2w)
        else:
            # use provided ray batch
            rays_o, rays_d = rays
    
        if self.use_viewdirs:
            # provide ray directions as input
            viewdirs = rays_d
            if c2w_staticcam is not None:
                # special case to visualize effect of viewdirs
                rays_o, rays_d = get_rays(self.H, self.W, self.K, c2w_staticcam)
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1,3]).float()
    
        sh = rays_d.shape # [..., 3]
        if self.ndc:
            # for forward facing scenes
            rays_o, rays_d = ndc_rays(self.H, self.W, self.K[0][0], 1., rays_o, rays_d)
    
        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()
    
        near, far = self.near * torch.ones_like(rays_d[...,:1]), self.far * torch.ones_like(rays_d[...,:1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        if self.use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)
    
        # Render and reshape
        all_ret = self.batchify_rays(rays, chunk, retraw=True)
        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)
    
        k_extract = ['rgb_map', 'disp_map', 'acc_map']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
        return ret_list + [ret_dict]
    
    # Given only camera poses and render the views
    def render(self, render_poses, chunk, render_factor=0):
    
        H, W, focal = self.H, self.W, self.focal
    
        if render_factor!=0:
            # Render downsampled for speed
            H = H//render_factor
            W = W//render_factor
            focal = focal/render_factor
    
        rgbs = []
        disps = []
    
        t = time.time()

        # TODO: this for-loop should be moved outside the class
        #       the input is a batch
        with torch.no_grad():
            for i, c2w in enumerate(tqdm(render_poses)):
                print(i, time.time() - t)
                t = time.time()
                rgb, disp, acc, _ = self(chunk=chunk, c2w=c2w[:3,:4])
                rgbs.append(rgb)
                disps.append(disp)
                if i==0:
                    print(rgb.shape, disp.shape)
    
                """
                if gt_imgs is not None and render_factor==0:
                    p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
                    print(p)
                """
                if i > 1:
                    break
    
        rgbs = torch.stack(rgbs, 0)
        disps = torch.stack(disps, 0)
    
        return rgbs, disps


def _xavier_init(model):
    """
    Performs the Xavier weight initialization.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
