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
import random
import numpy as np
import wandb

from utils.logger import setup_experiment
from algos.main import Model_Base

loss_train = []
loss_val = []

def run(cfg):
    """
    train CLIP
    params
    ------
    cfg: object
        config
    """
    # ---------- experiment prepearation ----------
    cwd, results_dir, device, run_wandb = setup_experiment(cfg)
    
    # ---------- train ----------
    #load train data
    D = Memory_Dataset(cfg, cwd, cfg.train.train_data_path[0], 
                       cfg.train.train_image_data_path[0], 
                       device=device)
    #load validation data
    D_val = Memory_Dataset(cfg, cwd, cfg.train.validation_data_path[0], 
                           cfg.train.validation_image_data_path[0], 
                           device=device)
    #set model
    model = Model_Base(cfg, device)
    train_data = model.CLIP.make_clipdata(D.image_path, D.texts)
    validation_data = model.CLIP.make_clipdata(D_val.image_path, D_val.texts)
    train_data = torch.utils.data.DataLoader(train_data, batch_size=cfg.train.batch_size, shuffle=True)
    validation_data = torch.utils.data.DataLoader(validation_data, batch_size=cfg.train.batch_size, shuffle=True)

    #train
    global loss_train, loss_val
    for itr in tqdm(range(0, cfg.train.train_iteration)):
        #train
        for i, data in enumerate(train_data):
            loss_train_tmp = model.train_CLIP(data, is_train=True)
        loss_train_tmp /= len(train_data)
        #validation
        if itr%cfg.train.validation_interval == 0:
            for i, data in enumerate(validation_data):
                loss_val_tmp = model.train_CLIP(data, is_train=False)
            loss_val_tmp /= len(validation_data)
        else:
            loss_val_tmp = None
        #save loss
        loss_train.append(loss_train_tmp)
        loss_val.append(loss_val_tmp)

        #wsave to wandb
        if cfg.main.wandb:
            run_wandb.log(data={f"loss/train": loss_train[-1]}, step=itr)
            if loss_val[-1] is not None:
                run_wandb.log(data={f"loss/validation": loss_val[-1]}, step=itr)
        
        #save model
        params = dict()
        params["model"] = model.CLIP.get_state_dict()
        params["optimizer"] = model.optimizer["CLIP"].state_dict()
        #save loss
        save_loss = dict()
        save_loss["train"] = loss_train
        save_loss["validation"] = loss_val
        params["loss"] = save_loss
        if (itr+1)%cfg.train.checkpoint_interval == 0:
            os.makedirs(f"{results_dir}/model", exist_ok=True)
            torch.save(params, f"{results_dir}/model/CLIP_{itr+1}.pt")
        else:
            os.makedirs(f"{results_dir}/latest", exist_ok=True)
            torch.save(params, f"{results_dir}/latest/CLIP_current.pt")
        



@hydra.main(config_path="config", config_name="config_train_clip_probvlm")
def main(cfg_raw : DictConfig) -> None:
    _cfg = copy.deepcopy(cfg_raw)
    _cfg.main.experiment_name = "train_CLIP"
    _cfg.main.tags = "CLIP"
    _cfg.train.use_clip_probvlm = True
    _cfg.train.condition_type = "text"
    _cfg.main.n_gpu = 1
    run(_cfg)

if __name__=="__main__":
    main()