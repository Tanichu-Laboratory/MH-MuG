import sys
import os
from pathlib import Path
module_path = os.path.join(Path().resolve(), '../')
sys.path.append(module_path)
os.environ['PYTHONPATH'] = module_path

import hydra
from omegaconf import DictConfig, OmegaConf
from utils.dataset import Memory_Dataset


import copy
from tqdm import tqdm
import torch
from torch.utils.data import DistributedSampler, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import wandb

from utils.logger import setup_experiment
from algos.main import Model_Base

def train(rank, n_gpu, cfg, results_dir, run_wandb, cwd, itr_start, loss_list_train, loss_list_val):
    dist.init_process_group("gloo", rank=rank, world_size=n_gpu)
    #gpu_id = ["cuda:2"]
    #device = device = torch.device(gpu_id[rank])
    device = device = torch.device("cuda:{}".format(rank) if torch.cuda.is_available() else "cpu")

    # ---------- Set up model ----------
    model = Model_Base(cfg, device)
    model = DDP(model, device_ids=[rank]).module

    # ---------- Set up dataloader----------
    #train
    D = Memory_Dataset(cfg, cwd, cfg.train.train_data_path[0], device=device)
    sampler = DistributedSampler(D.data, num_replicas=n_gpu, rank=rank, shuffle=True)
    train_loader = DataLoader(D.data, batch_size=cfg.train.batch_size, sampler=sampler)
    #validation
    D_val = Memory_Dataset(cfg, cwd, cfg.train.validation_data_path[0], device=device)
    sampler = DistributedSampler(D_val.data, num_replicas=n_gpu, rank=rank, shuffle=True)
    validation_loader = DataLoader(D_val.data, batch_size=cfg.train.batch_size, sampler=sampler)

    # ---------- training diffusion ----------
    #training
    for itr in tqdm(range(itr_start, cfg.train.train_iteration)):
        global_step = itr+1
        model.z_model.global_step = global_step #update global step
        #train
        loss_train = 0
        for batch_idx, data in enumerate(train_loader):
            loss_train += model.train_z_model(data)
        loss_train /= len(train_loader)
        loss_list_train.append(loss_train)
        #validation
        if global_step%cfg.train.validation_interval == 0:
            loss_val = 0
            for batch_idx, data in enumerate(validation_loader):
                loss_val += model.train_z_model(data, is_train=False)
            loss_val /= len(validation_loader)
            loss_list_val.append(loss_val)
        else:
            loss_list_val.append(None)
        #save wandb
        if cfg.main.wandb and rank==0:
            run_wandb.log(data={f"loss_diffusion/train": loss_train}, step=global_step)
            run_wandb.log(data={f"loss_diffusion/validation": loss_val}, step=global_step)
        
        #save model
        if rank==0 and global_step%cfg.train.checkpoint_interval == 0:
            #モデルのパラメータの保存
            params = dict()
            params["model"] = model.z_model.state_dict()
            params["optimizer"] = model.optimizer["z_model"].state_dict()
            #lossの保存
            params["loss"] = {"train": loss_list_train, "validation": loss_list_val}
            os.makedirs(f"{results_dir}/model", exist_ok=True)
            torch.save(params, f"{results_dir}/model/diffusion_{global_step}.pt")

def run(cfg):
    # ---------- Create experiment folder ----------
    #Get current working directory, results directory, and device
    cwd, results_dir, device, run_wandb = setup_experiment(cfg)

    # ---------- Set up dataset ----------
    #Train
    #D = Memory_Dataset(cfg, cwd, cfg.train.train_data_path[0], device=torch.device("cpu"))
    #Validation 
    #D_val = Memory_Dataset(cfg, cwd, cfg.train.validation_data_path[0], device=torch.device("cpu"))

    # ---------- Set up loss list ----------
    print("---------Set up loss list and itr start---------")
    loss_list_train = []
    loss_list_val = []
    model = Model_Base(cfg, device=torch.device("cpu"))

    #load loss
    itr_start = 0
    if cfg.vae.pretained_model_path is not None:
        #check if pretrain_mode is the same as train_mode; else, reset loss_list
        itr_start = len(model.loss_list["z_model"]["train"])
        loss_list_train = model.loss_list["z_model"]["train"]
        loss_list_val = model.loss_list["z_model"]["validation"]
        print(f"itr_start: {itr_start}")
        #save wandb
        if cfg.main.wandb:
            for i in range(itr_start):
                wandb.log(data={f"loss_diffusion/train": loss_list_train[i]}, step=i)
                if loss_list_val[i] is not None:
                    wandb.log(data={f"loss_diffusion/validation": loss_list_val[i]}, step=i)
        else:
            model.loss_list["z_model"]= {"train": [], "validation": []}

    # ---------- train ----------
    n_gpu = cfg.main.n_gpu
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '52388'
    print("---------Start training----------")
    mp.spawn(train,
        args=(n_gpu, cfg, results_dir, run_wandb, cwd, itr_start, loss_list_train, loss_list_val),
        nprocs=n_gpu,
        join=True)
        


@hydra.main(config_path="config", config_name="config_train_diffusion")
def main(cfg_raw : DictConfig) -> None:
    _cfg = copy.deepcopy(cfg_raw)
    _cfg.main.experiment_name = "train_dit"

    run(_cfg)

if __name__=="__main__":
    main()

