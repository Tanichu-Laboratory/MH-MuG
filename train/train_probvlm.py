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
import random
import numpy as np
import wandb

from utils.logger import setup_experiment
from algos.main import Model_Base

def train(rank, n_gpu, cfg, results_dir, run_wandb, cwd, itr_start, loss_train, loss_val):
    dist.init_process_group("gloo", rank=rank, world_size=n_gpu)
    #gpu_id = ["cuda:2"]
    #device = device = torch.device(gpu_id[rank])
    device = device = torch.device("cuda:{}".format(rank) if torch.cuda.is_available() else "cpu")
    # ---------- Set up model ----------
    model = Model_Base(cfg, device)
    model = DDP(model, device_ids=[rank]).module

    # ---------- Set up dataloader----------
    #train
    D = Memory_Dataset(cfg, cwd, cfg.train.train_data_path[0], 
                       cfg.train.train_image_data_path[0], 
                       device=device)
    D = model.CLIP.make_clipdata(D.image_path, D.texts)
    sampler = DistributedSampler(D, num_replicas=n_gpu, rank=rank, shuffle=True)
    train_loader = DataLoader(D, batch_size=cfg.train.batch_size, sampler=sampler)
    #validation
    D_val = Memory_Dataset(cfg, cwd, cfg.train.validation_data_path[0], 
                           cfg.train.validation_image_data_path[0], 
                           device=device)
    D_val = model.CLIP.make_clipdata(D_val.image_path, D_val.texts)
    sampler = DistributedSampler(D_val, num_replicas=n_gpu, rank=rank, shuffle=True)
    validation_loader = DataLoader(D_val, batch_size=cfg.train.batch_size, sampler=sampler)

    #train
    for itr in tqdm(range(itr_start, cfg.train.train_iteration)):
        global_step = itr+1
        #訓練データ
        loss_train_tmp = 0
        for i, data in enumerate(train_loader):
            loss_train_tmp += model.train_ProbVLM(data, is_train=True)
        loss_train_tmp /= len(train_loader)
        #検証データ
        if itr%cfg.train.validation_interval == 0:
            loss_val_tmp = 0
            for i, data in enumerate(validation_loader):
                loss_val_tmp += model.train_ProbVLM(data, is_train=False)
            loss_val_tmp /= len(validation_loader)
        else:
            loss_val_tmp = None
        #lossの保存
        loss_train.append(loss_train_tmp)
        loss_val.append(loss_val_tmp)

        #wandbに保存
        if cfg.main.wandb and rank==0:
            run_wandb.log(data={f"loss/train": loss_train[-1]}, step=global_step)
            if loss_val[-1] is not None:
                run_wandb.log(data={f"loss/validation": loss_val[-1]}, step=global_step)
        
        #モデルの保存
        if rank==0 and (itr+1)%cfg.train.checkpoint_interval == 0:
            #モデルのパラメータの保存
            params = dict()
            params["model"] = model.ProbVLM.state_dict()
            params["optimizer"] = model.optimizer["ProbVLM"].state_dict()
            #lossの保存
            save_loss = dict()
            save_loss["train"] = loss_train
            save_loss["validation"] = loss_val
            params["loss"] = save_loss
            os.makedirs(f"{results_dir}/model", exist_ok=True)
            torch.save(params, f"{results_dir}/model/ProbVLM_{global_step}.pt")

def run(cfg):
    """
    train probvlm
    params
    ------
    cfg: object
        config
    """
    # ---------- experiment preperation ----------
    cwd, results_dir, device, run_wandb = setup_experiment(cfg)
    
    # ---------- train ----------
    #set model
    model = Model_Base(cfg, device)

    #load loss
    loss_train = []
    loss_val = []
    itr_start = 0
    if cfg.clip_probvlm.ProbVLM["parameter_path"] is not None:
        loss_train = model.loss_list["ProbVLM"]["train"]
        loss_val = model.loss_list["ProbVLM"]["validation"]
        itr_start = len(loss_train)
        print(f"itr_start: {itr_start}")
        #save to wandb
        if cfg.main.wandb:
            for i in range(itr_start):
                wandb.log(data={f"loss/train": loss_train[i]}, step=i)
                if loss_val[i] is not None:
                    wandb.log(data={f"loss/validation": loss_val[i]}, step=i)
    # ---------- train ----------
    n_gpu = cfg.main.n_gpu
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '42355'
    print("---------Start training----------")
    mp.spawn(train,
        args=(n_gpu, cfg, results_dir, run_wandb, cwd, itr_start, loss_train, loss_val),
        nprocs=n_gpu,
        join=True)
    
        



@hydra.main(config_path="config", config_name="config_train_clip_probvlm")
def main(cfg_raw : DictConfig) -> None:
    _cfg = copy.deepcopy(cfg_raw)
    _cfg.main.experiment_name = "train_ProbVLM"
    _cfg.main.tags = "ProbVLM"
    _cfg.train.use_clip_probvlm = True
    _cfg.main.wandb = True
    run(_cfg)

if __name__=="__main__":
    main()