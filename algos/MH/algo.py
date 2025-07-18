import numpy as np
import os
import copy
import random
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
import torch.multiprocessing as mp
import clip
from peft import get_peft_model, LoraConfig
import cv2

class MH_naming():
    """
    MH_naming_game

    """
    def __init__(self, cfg, results_dir, model_A, model_B, texts):
        """
        Number of agents is 2, 0:A, 1:B
        params
        ------
        cfg: object config
        results_dir: str #result directory
        model_A, model_B: object #each agent's model
        texts: list[str]

        attributes
        ----------
        cfg: object #config
        result_dir: str #result directory
        n_gpu: int  #number of gpus
        mode: int #0:No com, 1:all accept
        acceptance: np.array (2, cfg.train.train_iteration) #acceptance rate
        like: np.array (2, cfg.train.train_iteration) #likelihood
        texts: list(str) #texts
        ..._tmp:  #temporary for multi-processing
        """
        self.cfg = cfg
        self.results_dir = results_dir
        self.mode = cfg.train.mode
        self.acceptance_rate = np.zeros((2, cfg.train.train_iteration))
        self.like = np.zeros((2, cfg.train.train_iteration))
        self.model = [model_A, model_B]
        self.texts = texts
        self.text_tmp = None
        self.img_conv_tmp = [None, None] #0:A, 1:B
        self.text_conv_tmp = None
        self.path_tmp = None #list(torch.tensor) #path of finetuning data A and B
    
    def w_to_imgconv(self, w_orig, clip_model):
        """
        Convert w to image_conv

        params
        ------
        w_orig: torch.tensor #w
        clip_model: object #clip model

        return
        ------
        imgconv: torch.tensor #image_conv
        """
        #set param
        w = copy.deepcopy(w_orig)
        img_dir_tmp = f"{self.results_dir}/img_tmp"
        os.makedirs(img_dir_tmp, exist_ok=True)
        img_pathes_tmp = []
        #convert to image
        for i in range(w.shape[0]):
            img_path_tmp = f"{img_dir_tmp}/{i}.jpg"
            img_pathes_tmp.append(img_path_tmp)
            img = w[i][0].detach().cpu().numpy()+1
            img[img<=0] = 0
            img[img>0] = 1
            img = img*255
            img = np.array([img, img, img]).transpose(2, 1, 0)
            cv2.imwrite(img_path_tmp, img)
        #convert to image_conv
        imgconv = clip_model.image_encoding(img_pathes_tmp)

        return imgconv

            

#---------------------- sampling diffusion ----------------------  
    def get_w_multi(self, rank):
        """
        Get w from model in multi-gpu
        Use 2 GPUs

        params
        ------
        rank: int
        itr: int
        """
        dist.init_process_group("gloo", rank=rank, world_size=2)

        # ------- Set up model ----------
        model = self.model[rank]

        #-------- Sampling W ----------
        agent_name = ["A", "B"]
        texts = self.text_tmp
        save_path = f"{self.results_dir}/sample/sample_current_{agent_name[rank]}.pt"
        os.makedirs(f"{self.results_dir}/sample", exist_ok=True)
        sample = model.sampling_diffusion(cond=texts)
        torch.save(sample, save_path)
    
    def get_w(self, agent_id):
        """
        Get w from model

        params
        ------
        agent_id: int   #0:agent A, 1:agent B
        itr: int #iteration number
        """
        model = self.model[agent_id]
        texts = self.text_tmp
        agent_name = ["A", "B"]
        save_path = f"{self.results_dir}/sample/sample_current_{agent_name[agent_id]}.pt"
        os.makedirs(f"{self.results_dir}/sample", exist_ok=True)
        sample = model.sampling_diffusion(cond=texts)
        torch.save(sample, save_path)    

#---------------------- acceptancerate ---------------------- 
    def cal_acceptancerate(self, img_conv_A_orig, img_conv_B_orig, text_conv_orig):
        """
        naming game speaker to listener
        params
        ------
        img_conv_A_orig, img_conv_B_orig: torch.tensor #image_conv(w conv)
        texts_orig: list[str] #text(ovsevation conv)

        return
        ------
        acc_A, acc_B: numpy.ndarray #acceptance rate
        like_A, like_B: numpy.ndarray #likelihood
        """
        #------- get param -------
        img_conv_A = copy.deepcopy(img_conv_A_orig)
        img_conv_B = copy.deepcopy(img_conv_B_orig)
        text_conv = copy.deepcopy(text_conv_orig)
        img_conv_A = img_conv_A.unsqueeze(0)
        img_conv_B = img_conv_B.unsqueeze(0)
        text_conv = text_conv.unsqueeze(0)
        #------- cal likelihood -------
        like_AA = self.model[0].ProbVLM.cal_likelihood(img_conv_A, text_conv)
        like_AB = self.model[1].ProbVLM.cal_likelihood(img_conv_A, text_conv)
        like_BA = self.model[0].ProbVLM.cal_likelihood(img_conv_B, text_conv)
        like_BB = self.model[1].ProbVLM.cal_likelihood(img_conv_B, text_conv)
        #------- cal acceptance rate -------
        like_A = like_AA+like_AB
        like_B = like_BA+like_BB
        acc_A = min(1, np.exp(like_BA-like_AA))
        acc_B = min(1, np.exp(like_AB-like_BB))

        return acc_A, acc_B, like_A, like_B
    
#---------------------- finetuning ----------------------
    def finetuning_multi(self, rank):
        """
        finetuning model in multi-gpu
        Use 2 GPUs

        params
        ------
        path_all: list(torch.tensor) #path of finetuning data A and B
        """
        dist.init_process_group("gloo", rank=rank, world_size=2)
        agent_id = rank
        path_all = self.path_tmp
        #------ make dataset -------
        model = self.model[agent_id]
        data = torch.load(path_all[rank], map_location=model.device)
        dataset = TensorDataset(data, self.text_conv_tmp.to(model.device))
        diffusion_loader = DataLoader(dataset, batch_size=self.cfg.train.batch_size_diffusion, shuffle=True)
        probvlm_dataset =TensorDataset(self.img_conv_tmp[rank], self.text_conv_tmp.to(model.device))
        probvlm_loader = DataLoader(probvlm_dataset, batch_size=self.cfg.train.batch_size_clip_probvlm, shuffle=True)
        #------ finetuning -------
        #Diffusion
        for epoch in tqdm(range(self.cfg.train.finetuning_diffusion_itr), desc="Diffusion finetuning"):
            for data in diffusion_loader:
                __ = model.train_z_model(data, is_train=True)
        #ProbVLM
        for epoch in tqdm(range(self.cfg.train.finetuning_probvlm_itr), desc="ProbVLM finetuning"):
            for data in probvlm_loader:
                __ = model.train_ProbVLM(data, is_train=True)
        self.save_parms(agent_id)


    def finetuning(self, data, agent_id):
        """
        finetuning model

        params
        ------
        data: torch.tensor #finetuning data
        agent_id: int   #0:agent A, 1:agent B
        """
        #------ make dataset -------
        model = self.model[agent_id]
        dataset = TensorDataset(data, self.text_conv_tmp)
        diffusion_loader = DataLoader(dataset, batch_size=self.cfg.train.batch_size_diffusion, shuffle=True)
        probvlm_dataset = TensorDataset(self.img_conv_tmp[agent_id].to(model.device), self.text_conv_tmp.to(model.device))
        probvlm_loader = DataLoader(probvlm_dataset, batch_size=self.cfg.train.batch_size_clip_probvlm, shuffle=True)
        #------ finetuning -------
        #Diffusion
        for epoch in tqdm(range(self.cfg.train.finetuning_diffusion_itr), desc="Diffusion finetuning"):
            for data in diffusion_loader:
                __ = model.train_z_model(data, is_train=True)
        #ProbVLM
        for epoch in tqdm(range(self.cfg.train.finetuning_probvlm_itr), desc="ProbVLM finetuning"):
            for data in probvlm_loader:
                __ = model.train_ProbVLM(data, is_train=True)
        self.save_parms(agent_id)

    
    def save_parms(self, agent_id):
        """
        Save current model parameters and optimizer parameters

        params
        ------
        itr: int
        agent_id: int   #0:agent A, 1:agent B
        """
        #set param
        model = self.model[agent_id]
        agent_name = ["A", "B"][agent_id]
        os.makedirs(f"{self.results_dir}/model_current/{agent_name}", exist_ok=True)
        #diffusion
        params_diffusion = dict()
        params_diffusion["model"] = model.z_model.state_dict()
        params_diffusion["optimizer"] = model.optimizer["z_model"].state_dict()
        torch.save(params_diffusion, f"{self.results_dir}/model_current/{agent_name}/diffusion.pt")
        #ProbVLM
        params_probvlm = dict()
        params_probvlm["model"] = model.ProbVLM.state_dict()
        params_probvlm["optimizer"] = model.optimizer["ProbVLM"].state_dict()
        torch.save(params_probvlm, f"{self.results_dir}/model_current/{agent_name}/ProbVLM.pt")
    
    def save_MH_log(self):
        """
        Save log of MH model
        """
        params_MH = dict()
        params_MH["acceptance_rate"] = self.acceptance_rate
        params_MH["likelihood"] = self.like
        #save
        os.makedirs(f"{self.results_dir}/model_current", exist_ok=True)
        torch.save(params_MH, f"{self.results_dir}/model_current//MH_current.pt")
    
#---------------------- train ----------------------
    def train(self, itr):
        """
        params
        ------
        itr: int

        return
        ------
        data_A, data_B: torch.tensor  #finetuning data in A, B on next step
        """
        #-------sampling W ------- 
        print("sampling")
        #sampling
        self.text_tmp = random.choices(self.texts, k=self.cfg.train.sampling_size)
        if self.cfg.main.n_gpu > 1:
            mp.spawn(self.get_w_multi, nprocs=2, args=())
        else:
            self.get_w(0)
            self.get_w(1)
        #load
        path_A = f"{self.results_dir}/sample/sample_current_A.pt"
        path_B = f"{self.results_dir}/sample/sample_current_B.pt"
        #path_A = "/raid/koki-sakurai/model/train/pretrained/sample/sample_classical_A.pt"
        #path_B = "/raid/koki-sakurai/model/train/pretrained/sample/sample_jazz_B.pt"
        w_A = torch.load(path_A, map_location=torch.device("cpu"))["midi"]
        w_B = torch.load(path_B, map_location=torch.device("cpu"))["midi"]
        self.text_tmp = torch.load(path_A, map_location=torch.device("cpu"))["text"]

        #------- cal acceptancerate -------
        print("cal_acceptancerate")
        num_samples = len(w_A)
        acc_A = np.zeros(num_samples)
        acc_B = np.zeros(num_samples)
        like_A = np.zeros(num_samples)
        like_B = np.zeros(num_samples)
        data_A = torch.zeros((w_A.shape)) #finetuning data in A on next step
        data_B = torch.zeros((w_A.shape)) #finetuning data in B on next step
        img_conv_A = self.w_to_imgconv(w_A, self.model[0].CLIP)
        img_conv_B = self.w_to_imgconv(w_B, self.model[1].CLIP)
        img_conv_A_tmp = torch.zeros((img_conv_A.shape))
        img_conv_B_tmp = torch.zeros((img_conv_B.shape))
        self.text_conv_tmp = clip.tokenize(self.text_tmp)
        for i in tqdm(range(num_samples)):
            acc_A[i], acc_B[i], like_A[i], like_B[i] = self.cal_acceptancerate(img_conv_A[i], img_conv_B[i], self.text_conv_tmp[i])
            #accept or not
            if np.random.rand() < acc_A[i]:
                data_A[i] = torch.tensor(w_B[i], dtype=torch.float32, device=self.model[0].device)
                img_conv_A_tmp[i] = img_conv_B[i]
            else:
                data_A[i] = torch.tensor(w_A[i], dtype=torch.float32, device=self.model[0].device)
                img_conv_A_tmp[i] = img_conv_A[i]
            if np.random.rand() < acc_B[i]:
                data_B[i] = torch.tensor(w_A[i], dtype=torch.float32, device=self.model[1].device)
                img_conv_B_tmp[i] = img_conv_A[i]
            else:
                data_B[i] = torch.tensor(w_B[i], dtype=torch.float32, device=self.model[1].device)
                img_conv_B_tmp[i] = img_conv_B[i]
        self.img_conv_tmp[0] = img_conv_A_tmp.to(self.model[0].device)
        self.img_conv_tmp[1] = img_conv_B_tmp.to(self.model[1].device)
        #------- save log -------
        self.acceptance_rate[0][itr] = np.mean(acc_A)
        self.acceptance_rate[1][itr] = np.mean(acc_B)
        self.like[0][itr] = np.mean(like_A)
        self.like[1][itr] = np.mean(like_B)

        #------- finetuning-------
        os.makedirs(f"{self.results_dir}/sample", exist_ok=True)
        path_A_accepted = f"{self.results_dir}/sample/sample_current_accepted_A.pt"
        path_B_accepted = f"{self.results_dir}/sample/sample_current_accepted_B.pt"
        torch.save(data_A, path_A_accepted)
        torch.save(data_B, path_B_accepted)
        if self.cfg.main.n_gpu > 1:
            print("finetuning")
            self.path_tmp = [path_A_accepted, path_B_accepted]
            mp.spawn(self.finetuning_multi, nprocs=2, args=())
        else:
            print("finetuning A")
            self.finetuning(data_A, 0)
            print("finetuning B")
            self.finetuning(data_B, 1)
        self.save_MH_log()
        