import os

from os.path import join as ospj
from os.path import expanduser
from munch import Munch as mch
from tqdm import tqdm_notebook
import numpy as np
import torch
import torch.nn as nn

from algos.ProbVLM.networks import BayesCap_for_CLIP
from algos.ProbVLM.losses import TempCombLoss

class ProbVLM(nn.Module):
    """
    ProbVLM 

    params
    ------
    cfg: dict #config
    CLIP: torch.nn.Module #CLIP model
    device: torch.device #device
    """
    def __init__(self, cfg, CLIP, device):
        super().__init__()
        self.cfg = cfg
        self.CLIP = CLIP
        self.device = device
        self.net = BayesCap_for_CLIP(
            inp_dim=self.cfg.clip_probvlm.ProbVLM["inp_dim"],
            out_dim=self.cfg.clip_probvlm.ProbVLM["out_dim"],
            hid_dim=self.cfg.clip_probvlm.ProbVLM["hid_dim"],
            num_layers=self.cfg.clip_probvlm.ProbVLM["num_layers"],
            p_drop=self.cfg.clip_probvlm.ProbVLM["p_drop"],
        ).to(device) #ProbVLM network
        self.loss = TempCombLoss().to(device)
        self.T1 = 1e0
        self.T2 = 5e-2
        self.cross_modal_lambda=1e-4

    def forward(self, img, text):
        """
        params
        ------
        img: torch.Tensor #conved
        text: torch.Tensor #conved
        """
        #encoding
        with torch.no_grad():
            xfI = self.CLIP.model.encode_image(img).to(self.device)
            xfT = self.CLIP.model.encode_text(text).to(self.device)
        #ProbVLM
        (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta) = self.net(xfI, xfT)
        loss_i = self.loss(img_mu, img_1alpha, img_beta, xfI, T1=self.T1, T2=self.T2)
        loss_t = self.loss(txt_mu, txt_1alpha, txt_beta, xfT, T1=self.T1, T2=self.T2)
        #cross modal terms
        loss_i4t = self.loss(img_mu, img_1alpha, img_beta, xfT, T1=self.T1, T2=self.T2)
        loss_t4i = self.loss(txt_mu, txt_1alpha, txt_beta, xfI, T1=self.T1, T2=self.T2)
        loss = loss_i + loss_t + self.cross_modal_lambda*(loss_i4t + loss_t4i)

        return loss
    
    @torch.no_grad()
    def cal_likelihood(self, img, text):
        """
        params
        ------
        img: torch.Tensor #conved
        text: torch.Tensor #conved
        Use_VAE: bool #use VAE

        return
        ------
        liklihood: torch.Tensor #cross modal liklihood
        """
        #set loss reduction to sum
        self.loss = TempCombLoss(reduction="sum").to(self.device)
        img = img.to(self.device)
        text = text.to(self.device)
        #encoding
        with torch.no_grad():
            xfI = self.CLIP.model.encode_image(img).to(self.device)
            xfT = self.CLIP.model.encode_text(text).to(self.device)
        #ProbVLM
        (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta) = self.net(xfI, xfT)
        #cross modal terms
        loss_i4t = self.loss(img_mu, img_1alpha, img_beta, xfT, T1=self.T1, T2=self.T2)
        liklihood = loss_i4t

        #set loss reduction to mean
        self.loss = TempCombLoss(reduction="mean").to(self.device)

        return liklihood.to("cpu")
    
    @torch.no_grad()
    def get_text_mean(self, text_enc):
        """
        params
        ------
        text_enc: torch.Tensor #encoded text by CLIP

        return
        ------
        text_mu: torch.Tensor #mean of ProbVLM
        """
        dummy = torch.zeros_like(text_enc)
        (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta) = self.net(dummy, text_enc)

        return txt_mu

        



