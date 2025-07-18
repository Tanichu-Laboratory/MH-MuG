from math import sqrt
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from einops import rearrange, reduce
from torch import Tensor

from tqdm import tqdm

from utils.params import (extract,
                          extract_2,
                          BetaSchedule)
from algos.U_Net.u_net_std import UNetModel #https://github.com/aik2mlj/polyffusion
from algos.Diffusion.dit import DiTRotary_XL_8, DiTRotary_XL_8_textcond
from utils.loss import mse_loss_mean, zero_one_loss_mean
    
#--------- Diffusion -------------
class Diffusion(nn.Module):
    """
    Algorythm of Diffusion model(DDPM) 

    params
    ------
    cfg: object
        config
    device: torch.device
        using device
    """
    def __init__(self, cfg, device):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.num_steps = cfg.diffusion.num_steps #num steps
        #parameters related α, β
        self.beta_schedule = BetaSchedule(cfg.diffusion.beta["start"],
                                                    cfg.diffusion.beta["end"],
                                                    cfg.diffusion.num_steps,
                                                    device) #β
        self.beta = self.beta_schedule.betas #β
        self.alpha = 1.0 - self.beta #α
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0) #𝛼¯
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod) #√𝛼¯

        #sampling parameters
        with torch.no_grad():
            alpha_cumprod_prev = torch.cat([self.alpha_cumprod.new_tensor([1.]), self.alpha_cumprod[:-1]]) #𝛼¯_t-1
            self.sqrt_1m_alpha_cumprod = (1. - self.alpha_cumprod)**.5 #√(1-𝛼¯)
            self.sqrt_recip_alpha_cumprod = self.alpha_cumprod**-.5 #√(1/𝛼¯)
            self.sqrt_recip_m1_alpha_cumprod = (1 / self.alpha_cumprod - 1)**.5 #√(1/(1-𝛼¯))
            variance = self.beta * (1. - alpha_cumprod_prev) / (1. - self.alpha_cumprod)
            self.log_var = torch.log(torch.clamp(variance, min=1e-20))
            self.mean_x0_coef = self.beta * (alpha_cumprod_prev**.5) / (1. - self.alpha_cumprod)
            self.mean_xt_coef = (1. - alpha_cumprod_prev) * ((1 - self.beta)** 0.5) / (1. - self.alpha_cumprod)

        
        #setting network
        self.network = cfg.diffusion.denoising_network
        if self.network == "u_net":
            self.u_net_config = cfg.diffusion.u_net
            self.net = UNetModel(
                self.u_net_config["channel_init"],
                self.u_net_config["channel_init"],
                self.u_net_config["channel"],
                self.u_net_config["num_blocks"],
                self.u_net_config["attention_levels"],
                self.u_net_config["multiple_layer"],
                self.u_net_config["n_head"],
                self.u_net_config["tf_layer"],
                self.u_net_config["d_cond"]
                            ).to(device)
        elif self.network == "dit":
            self.dit_config = cfg.diffusion.dit
            if self.cfg.train.condition_type == "text":
                self.net = DiTRotary_XL_8_textcond(
                    input_size=self.dit_config["input_size"],
                    in_channels=self.dit_config["in_channels"],
                    num_classes=self.dit_config["num_classes"],
                    class_dropout_prob=self.dit_config["class_dropout_prob"],
                    learn_sigma=self.dit_config["learn_sigma"],
                ).to(device)
            else:
                self.net = DiTRotary_XL_8(
                    input_size=self.dit_config["input_size"],
                    in_channels=self.dit_config["in_channels"],
                    num_classes=self.dit_config["num_classes"],
                    class_dropout_prob=self.dit_config["class_dropout_prob"],
                    learn_sigma=self.dit_config["learn_sigma"],
                ).to(device)
        else:
            raise ValueError(f"denoising_network: {self.network} is not defined")
    
    def forward(self, x, is_train=True, cond=None):
        """
        x: Tensor(batch_size, channel*modalities, length, pitch)
        is_train: bool 
        cond: Tensor(batch_size, 1, d_cond) condition
        """
        #obtarin parameters
        batch_size = x.shape[0] 
        t = torch.randint(0, self.cfg.diffusion.num_steps, (batch_size,), device=self.device, dtype=torch.long)#step information
        #obtain parameter at t
        alpha_cumprod_t = extract_2(self.alpha_cumprod, t, batch_size, self.device) #𝛼¯_t
        sqrt_alpha_cumprod_t = extract_2(self.sqrt_alpha_cumprod, t, batch_size, self.device) # √𝛼¯_t

        #add noise (q(x_t|x_0)
        noise = torch.randn_like(x)
        mean = sqrt_alpha_cumprod_t * x
        var = 1.0 - alpha_cumprod_t
        x_t = mean + (var**0.5) * noise

        #predict noise
        if cond is None and self.network == "u_net":
            cond = -torch.ones((batch_size, 1, self.u_net_config["d_cond"]), device=self.device)
            cond = cond.to(x_t.dtype)
        if self.network == "u_net":
            noise_pred = self.net(x_t, t, cond) #u-net
        elif self.network == "dit":
            if self.cfg.train.condition_type == "text":
                y = cond
            else:
                y = torch.tensor([self.dit_config["num_classes"]] * x.shape[0], device=x.device)
            noise_pred = self.net(x_t, t, y) #DiT
        else:
            raise ValueError(f"denoising_network: {self.network} is not defined")

        #caluculate loss
        loss = F.mse_loss(noise, noise_pred)

        return loss
    
    @torch.no_grad()
    def forward_val(self, x, is_train=False, cond=None, t_init=None):
        """
        x: Tensor(batch_size, channel*modalities, length, pitch)
        is_train: bool
        cond: Tensor(batch_size, 1, d_cond) condition
        t_init: int #step information
        """
        #obtain parameteres
        batch_size = x.shape[0] 
        t = torch.randint(0, self.cfg.diffusion.num_steps, (batch_size,), device=self.device, dtype=torch.long)#step information
        if t_init is not None:
            t = (torch.ones(batch_size, dtype=torch.long)*t_init).to(self.device)
        #obtain t_informations
        alpha_cumprod_t = extract_2(self.alpha_cumprod, t, batch_size, self.device) #𝛼¯_t
        sqrt_alpha_cumprod_t = extract_2(self.sqrt_alpha_cumprod, t, batch_size, self.device) # √𝛼¯_t

        #add noise q(x_t|x_0)
        noise = torch.randn_like(x)
        mean = sqrt_alpha_cumprod_t * x
        var = 1.0 - alpha_cumprod_t
        x_t = mean + (var**0.5) * noise

        #predict noise
        if cond is None and self.network == "u_net":
            cond = -torch.ones((batch_size, 1, self.u_net_config["d_cond"]), device=self.device)
            cond = cond.to(x_t.dtype)
        if self.network == "u_net":
            noise_pred = self.net(x_t, t, cond) #u-net
        elif self.network == "dit":
            if self.cfg.train.condition_type == "text":
                y = cond
            else:
                y = torch.tensor([self.dit_config["num_classes"]] * x.shape[0], device=x.device)
            noise_pred = self.net(x_t, t, y) #DiT
        else:
            raise ValueError(f"denoising_network: {self.network} is not defined")

        #caluculate loss
        if t_init is not None:
            loss = F.mse_loss(noise, noise_pred, reduction="none")
        else:
            loss = F.mse_loss(noise, noise_pred)

        return loss
    
    @torch.no_grad()
    def sample(self, 
               x, t, step, 
               sqrt_recip_alpha_cumprod,
               sqrt_recip_m1_alpha_cumprod,
               cond=None):
        """
        x: Tensor(batch_size, channel*modalities, length, pitch)
        t: Tensor(batch_size, )
        step: int
        sqrt_recip_alpha_cumprod: Tensor(batch_size, 1, 1, 1)
        sqrt_recip_m1_alpha_cumprod: Tensor(batch_size, 1, 1, 1)
        cond: Tensor(batch_size, 1, d_cond) condition
        """
        batch_size = x.shape[0]
        #denoising
        e_t = self.net(x, t, cond)
        x0 = sqrt_recip_alpha_cumprod * x - sqrt_recip_m1_alpha_cumprod * e_t
        mean_x0_coef = x.new_full((batch_size, 1, 1, 1), self.mean_x0_coef[step])
        mean_xt_coef = x.new_full((batch_size, 1, 1, 1), self.mean_xt_coef[step])
        mean = mean_x0_coef * x0 + mean_xt_coef * x
        log_var = x.new_full((batch_size, 1, 1, 1), self.log_var[step])
        #not add noise at t=0
        if step == 0:
            noise = 0
        else:
            noise = torch.randn(x.shape, device=self.device)
        x_prev = mean + (0.5 * log_var).exp() * noise

        return x_prev
            
    
    @torch.no_grad()
    def sampler(self, x, cond=None, decoder=None):
        """
        sampling
        x: Tensor(batch_size, channel*modalities, length, pitch)
        cond: Tensor(batch_size, 1, d_cond) condition
        decoder: object #decoder for scg sampling
        """
        #obtain parameters
        batch_size = x.shape[0]
        if cond is None and self.network == "u_net":
            cond = -torch.ones((batch_size, 1, self.u_net_config["d_cond"]), device=self.device)
            cond = cond.to(x.dtype)
        elif cond is None and self.cfg.train.condition_type == "class" and self.network == "dit":
            cond = torch.tensor([self.dit_config["num_classes"]] * x.shape[0], device=x.device)

        #denoising sampling
        bar = tqdm(total=self.num_steps, desc="sampling")
        for step in reversed(range(0, self.num_steps)):
            #parameter at t
            t = x.new_full((batch_size, ), step, dtype=torch.long)
            sqrt_recip_alpha_cumprod = x.new_full(
                (batch_size, 1, 1, 1), self.sqrt_recip_alpha_cumprod[step]
            ) #√(1/𝛼¯_t)
            sqrt_recip_m1_alpha_cumprod = x.new_full(
                (batch_size, 1, 1, 1), self.sqrt_recip_m1_alpha_cumprod[step]
            ) #√(1/(1-𝛼¯_t))

            #sampling
            x = self.sample(x, t, step, sqrt_recip_alpha_cumprod, sqrt_recip_m1_alpha_cumprod, cond)
            bar.update(1)
        return x

    #save parameters
    def get_state_dict(self):
        return self.state_dict()
