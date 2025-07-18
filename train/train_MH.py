import sys
import os
from pathlib import Path

import torch.utils
module_path = os.path.join(Path().resolve(), '../')
sys.path.append(module_path)
os.environ['PYTHONPATH'] = module_path

import hydra
from omegaconf import DictConfig, OmegaConf
from utils.dataset import Memory_Dataset
from utils.process_sign import adjust_sign


import copy
from tqdm import tqdm
import torch
import numpy as np
import wandb

from utils.logger import setup_experiment
from algos.main import Model_Base
from algos.MH.algo import MH_naming

def save_parms(results_dir, itr, model, agent_name):
        """
        Save model parameters and optimizer parameters

        params
        ------
        itr: int
        model: object
        agent_name: str
        """
        #diffusion
        params_diffusion = dict()
        params_diffusion["model"] = model.z_model.state_dict()
        params_diffusion["optimizer"] = model.optimizer["z_model"].state_dict()
        os.makedirs(f"{results_dir}/model_{itr}/{agent_name}", exist_ok=True)
        torch.save(params_diffusion, f"{results_dir}/model_{itr}/{agent_name}/diffusion_{itr}.pt")
        #probvlm
        params_probvlm = dict()
        params_probvlm["model"] = model.ProbVLM.state_dict()
        params_probvlm["optimizer"] = model.optimizer["ProbVLM"].state_dict()
        os.makedirs(f"{results_dir}/model_{itr}/{agent_name}", exist_ok=True)
        torch.save(params_probvlm, f"{results_dir}/model_{itr}/{agent_name}/probvlm_{itr}.pt")

def run(cfg):
    # ---------- Create experiment folder ----------
    #Get current working directory, results directory, and device
    cwd, results_dir, device, run_wandb = setup_experiment(cfg)
    if cfg.main.n_gpu > 1:
        device_A = torch.device("cuda:1")
        device_B = torch.device("cuda:2")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '62346'
    else:
        device_A = device
        device_B = device
    
    # ---------- Load text ----------
    print("---------- Load text data ----------")
    data_types = ["train", "val", "test"]
    texts = []
    for data_type in data_types:
        for dir in [cfg.train.dataset_dir_A, cfg.train.dataset_dir_B]:
            path = f"{dir}/{data_type}"
            data = Memory_Dataset(cfg, cwd, path, device=device)
            texts += (data.texts)


    # ---------- Set up model ----------
    print("---------- Set up model ----------")
    cfg.vae.pretained_model_path = cfg.train.A["vae_path"]
    cfg.diffusion.pretained_model_path = cfg.train.A["diffusion_path"]
    cfg.clip_probvlm.CLIP["parameter_path"] = cfg.train.A["clip_path"]
    cfg.clip_probvlm.ProbVLM["parameter_path"] = cfg.train.A["probvlm_path"]
    model_A = Model_Base(cfg, device_A)
    cfg.vae.pretained_model_path = cfg.train.B["vae_path"]
    cfg.diffusion.pretained_model_path = cfg.train.B["diffusion_path"]
    cfg.clip_probvlm.CLIP["parameter_path"] = cfg.train.B["clip_path"]
    cfg.clip_probvlm.ProbVLM["parameter_path"] = cfg.train.B["probvlm_path"]
    model_B = Model_Base(cfg, device_B)
    MH_model = MH_naming(cfg, results_dir, model_A, model_B, texts)
    itr_start = 0
    #load model
    if cfg.train.mh_model_path is not None:
        print(f"load MH model from {cfg.train.mh_model_path}")
        mh_params = torch.load(cfg.train.mh_model_path)
        for i in range(itr_start, cfg.train.train_iteration):
            accept_a = mh_params["acceptance_rate"][0][i]
            accept_b = mh_params["acceptance_rate"][1][i]
            like_a = mh_params["likelihood"][0][i]
            like_b = mh_params["likelihood"][1][i]
            if not sum([accept_a, accept_b]) == 0:
                itr_start = i+1
                MH_model.acceptance_rate[0][i] = accept_a
                MH_model.acceptance_rate[1][i] = accept_b
                if cfg.main.wandb:
                    run_wandb.log(data={f"accepted_A": accept_a}, step=i)
                    run_wandb.log(data={f"accepted_B": accept_b}, step=i)
                    run_wandb.log(data={f"likelihood_A": like_a}, step=i)
                    run_wandb.log(data={f"likelihood_B": like_b}, step=i)

    # ---------- MH training ----------
    print(f"itr_start:{itr_start}")
    for itr in tqdm(range(itr_start, cfg.train.train_iteration), desc="MH training"):
        MH_model.train(itr)

        #save wandb
        if cfg.main.wandb:
            run_wandb.log(data={f"accepted_A": MH_model.acceptance_rate[0][itr]}, step=itr)
            run_wandb.log(data={f"accepted_B": MH_model.acceptance_rate[1][itr]}, step=itr)
            run_wandb.log(data={f"likelihood_A": MH_model.like[0][itr]}, step=itr)
            run_wandb.log(data={f"likelihood_B": MH_model.like[1][itr]}, step=itr)
        
        #save model
        if (itr+1)%cfg.train.checkpoint_interval == 0:
            #モデルのパラメータの保存
            save_parms(results_dir, (itr+1), model_A, "A")
            save_parms(results_dir, (itr+1), model_B, "B")
            params_MH = dict()
            params_MH["acceptance_rate"] = MH_model.acceptance_rate
            params_MH["likelihood"] = MH_model.like
            #save
            os.makedirs(f"{results_dir}/model_{itr+1}", exist_ok=True)
            torch.save(params_MH, f"{results_dir}/model_{itr+1}/MH_{itr+1}.pt")
    



#configのファイル名を変える場合はconfig_nameを変更
@hydra.main(config_path="config", config_name="config_train_MH")
def main(cfg_raw : DictConfig) -> None:
    _cfg = copy.deepcopy(cfg_raw)
    _cfg.main.experiment_name = "MH_naming_game"
    _cfg.train.condition_type = "text"
    _cfg.diffusion.sampling_size = _cfg.train.sampling_size
    if _cfg.main.n_gpu > 1:
        _cfg.main.n_gpu = 2
    run(_cfg)

if __name__=="__main__":
    main()

