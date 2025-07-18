import os

import numpy as np
import torch

import datetime
import subprocess

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf

import wandb

def get_base_folder_name(cwd=".", experiment_name="."):
    """
    make folder and obtain the folder path to save the experimental results
    params
    ------
    cwd: str
        current folder path
    experiment_name: str
    
    returns
    --------
    base_folder_name: str
        folder path to save the experimental results
    run_name: str
        wandb run name
    """
    #initialize
    dt_now = datetime.date.today()
    count = 0

    #make folder and obtain folder path
    while(True):
        base_folder_name = f"{cwd}/results/{experiment_name}/{dt_now}/run_{count}"
        if not os.path.exists(base_folder_name):
            print(f"base_folder_name: {base_folder_name}")
            break
        else:
            count += 1
    run_name = f"{experiment_name}/{dt_now}/run_{count}"
    os.makedirs(base_folder_name, exist_ok=True)
    return base_folder_name, run_name

def cfg_to_dict(cfg):
    """
    conver config to dict or list

    params
    ------
    cfg: object
        config.yalm
    
    returns
    -------
    cfg_dict: dict(list)
    """
    if type(cfg) == DictConfig:
        cfg_dict = dict()
        for key in cfg.keys():
            cfg_dict[key] = cfg_to_dict(cfg[key])
        return cfg_dict
    elif type(cfg) == ListConfig:
        return list(cfg)
    else:
        return cfg

def setup_experiment(cfg):
    """
    params
    ------
    cfg: object
        config.yalm
    
    returns
    -------
    cwd: str #current folder path
    resurt_dir: str
    device: torch.device #using device
    run_wandb: object #current wandb
    """
    print("Setup experiment")
    if cfg.main.experiment_name == None:
        print("Please set experiment_name")
        quit()
    #folder path to save the experimental results
    cwd = hydra.utils.get_original_cwd()
    results_dir, run_name = get_base_folder_name(cwd, cfg.main.experiment_name)

    #save config
    file_name_cfg = '{}/hydra_config.yaml'.format(results_dir)
    OmegaConf.save(cfg, file_name_cfg)

    #display config
    print(' ' * 5 + 'Options')
    for k, v in cfg.items():
        print(' ' * 5 + k)
        for k2, v2 in v.items():
            print(' ' * 10 + k2 + ': ' + str(v2))
    
    #set device
    if torch.cuda.is_available() and not cfg.main.disable_cuda:
        print(f"Using {cfg.main.n_gpu} cuda GPUs")
        device = torch.device(cfg.main.device)
        torch.cuda.manual_seed(cfg.train.seed)  
    else:
        print("Using CPU")
        device = torch.device("cpu")

    #set wandb
    run_wandb = None
    if cfg.main.wandb:
        run_wandb =  wandb.init(name=run_name, 
                        project=cfg.main.experiment_name, 
                        config=cfg_to_dict(cfg), 
                        tags=[cfg.main.tags],
                        group="multi_gpu")
    
    return cwd, results_dir, device, run_wandb