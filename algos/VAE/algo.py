import importlib
import torch
import torch.nn as nn
from omegaconf import OmegaConf

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_model(name, ckpt):
    config = OmegaConf.load(name)
    model = instantiate_from_config(config.model)
    if ckpt is not None:
        model.init_from_ckpt(ckpt) # load_state_dict(mc['state_dict'])
    model.eval()
    return model


def load_data(name):
    config = OmegaConf.load(name)
    data = instantiate_from_config(config.data)
    return data

class VAE(nn.Module):
    """
    setting VAE model from taming

    params
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.optimizer_idx = 0
        if self.cfg.train.train_discriminator:
            self.optimizer_idx = 1
        self.scale_factor = self.cfg.vae.scale_factor

        #load model
        self.model = load_model(cfg.vae.model_config_path, cfg.vae.ckpt_path)
    
    def forward(self, x):
        loss = self.model.training_step(x, 0, self.optimizer_idx)
        return loss
    
    @torch.no_grad()
    def forward_val(self, x):
        with torch.no_grad():
            loss = self.model.training_step(x, 0, self.optimizer_idx)
        return loss
    
    @torch.no_grad()
    def get_z(self, x):
        posterior = self.model.encode(x)
        z = posterior.mode()
        z *= self.scale_factor
        return z
    
    @torch.no_grad()
    def decode_z(self, z, scale_factor=True):
        if scale_factor:
            z = z/self.scale_factor
        dec = self.model.decode(z)
        return dec
    
    @torch.no_grad()
    def get_mu_sigma(self, x):
        posterior = self.model.encode(x)
        mu = posterior.mean
        sigma = posterior.var
        sample = posterior.sample()
        return sample, mu, sigma
    
    @torch.no_grad()
    def sampling(self, x, scale_factor=True):
        posterior = self.model.encode(x)
        z = posterior.sample()
        if scale_factor:
            z = z/self.scale_factor
        dec = self.model.decode(z)
        return dec

