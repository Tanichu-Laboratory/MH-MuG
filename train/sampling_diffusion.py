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

def run(cfg):
    """
    params
    ------
    cfg: object #config
    """
    # ---------- set up experiment ----------
    __, results_dir, device, __ = setup_experiment(cfg)

    # ---------- load text ----------
    path = "/raid/koki-sakurai/model/train/pretrained/sample/sample_text.txt"
    cond = []
    """
    with open(path, "r") as f:
        for line in f:
            cond.append(line.strip())
    """
    for i in range(100):
        cond.append("Emerging in the late 1920s, this energetic piano style blends African American blues with jazz rhythms. Originating in the Southern United States, it gained popularity in urban centers like Chicago and New York. Characterized by its driving bass patterns, it influenced swing and rock 'n' roll music.")
    
    # ---------- set model----------
    model = Model_Base(cfg, device)

    # ---------- sampling ----------
    if cfg.diffusion.pretained_model_path is not None:
        print("sampling")
        sample = model.sampling_diffusion(cond=cond)
        os.makedirs(f"{results_dir}/sample", exist_ok=True)
        torch.save(sample, f"{results_dir}/sample/sample.pt")

    else:
        NotImplementedError("model_path is None")
        


@hydra.main(config_path="config", config_name="config_train_diffusion")
def main(cfg_raw : DictConfig) -> None:
    _cfg = copy.deepcopy(cfg_raw)
    _cfg.main.experiment_name = "sampling_diffusion"
    _cfg.main.n_gpu = 1
    _cfg.main.wandb = False
    run(_cfg)

if __name__=="__main__":
    main()

