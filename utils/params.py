from typing import Any
import torch
from torch import nn, optim
from torch import Tensor
from torch.nn import functional as F

#--------- extract ----------
def extract_2(x, t, batch_size, device):
    """
    extract 4D data at t

    params
    ------
    x: torch.Tensor
    t: torch.Tensor
    batch_size: int
    device: torch.device

    return
    ------
    x_t: torch.Tensor
    """
    out = x.gather(-1, t)
    return out.reshape(batch_size, 1, 1, 1).to(device)

def extract(x, t, batch_size, device):
    """
    extract 3D data at t

    params
    ------
    x: torch.Tensor
    t: torch.Tensor
    batch_size: int
    device: torch.device

    return
    ------
    x_t: torch.Tensor
    """
    out = x.gather(-1, t)
    return out.reshape(batch_size, 1, 1).to(device)

#--------- beta ----------
class BetaSchedule():
    """
    distribution of β

    params
    ------
    beta_start: float
    beta_end: float
    steps: int
    device: torch.device

    attributes
    ----------
    betas: torch.tensor
    """
    def __init__(self,
                 beta_start: float,
                 beta_end: float,
                 steps: int=100,
                 device: torch.device = torch.device("cpu")):
        self.betas = torch.linspace(beta_start, beta_end, steps).to(device)
        
