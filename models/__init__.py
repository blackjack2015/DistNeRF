from models.nerf import NeRF
from models.mipnerf import MipNeRF


model_dict = {
    'nerf': NeRF,
    'MipNeRF': MipNeRF
}

def get_model(model_name, config):

    # model = model_dict[model_name](**config)
    if model_name == 'nerf':
        model = NeRF(multires=config.multires, multires_views=config.multires_views,
                     i_embed=config.i_embed, depth=config.netdepth, hidden=config.netwidth,
                     skips=[4], use_viewdirs=config.use_viewdirs, lindisp=config.lindisp,
                     N_samples=config.N_samples, N_importance=config.N_importance,
                     white_bkgd=config.white_bkgd,
                     ).cuda()
    else:
        model = MipNeRF(device=torch.device("cuda"))
    return model


