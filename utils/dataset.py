import os

import numpy as np
import random
import glob
import torch
import clip

import hydra
from tqdm import tqdm
from omegaconf import DictConfig, ListConfig, OmegaConf

def np_to_tensor(data, device=torch.device("cpu")):
    """
    convert numpy data to tensor data

    params
    ------
    data: numpy.ndarray
        numpy data
    device: torch.device
        using device

    return
    -------
    data: tensor
        tensor data
    """
    if not torch.is_tensor(data):
        return torch.tensor(data, dtype=torch.float32, device=device)
    else:
        return data

class Memory_Dataset:
    """
    setting dataset

    params
    ------
    cfg: object
        config
    cwd: str
        current folder path
    dataset_path: str
    image_path: str
    device: torch.device
        using device
        
    attributions
    -------------
    cfg: object
    cwd: str
    dataset_path: str
    file_names: list
    inputs: dict[tensor]
    data: tensor #converted inputs to tensor
    loader: data_loader
    image_path: list
    texts: list
    device: torch.device
    """
    def __init__(self,
                 cfg,
                 cwd,
                 dataset_path,
                 image_path=None,
                 device=torch.device("cpu")
                 ):
        self.cfg = cfg
        self.cwd = cwd
        self.dataset_path = dataset_path
        self.device = device
        self.file_names = self.get_file_names()
        #initialize input data
        self.inputs = dict()
        for name in self.cfg.train.input_names:
            self.inputs[name] = torch.empty(
                (len(self.file_names), *self.cfg.train.input_shapes[name]),
                dtype=torch.float32
            )
        self.data = None
        self.loader = None
        self.image_path = []
        if not image_path is None:
            self.image_path = sorted(glob.glob(os.path.join(self.cwd, image_path, "*.jpg")))
        self.labels = torch.empty(
                len(self.file_names),
                dtype=torch.float32
            )
        self.texts = []
        self.load_dataset()


    #------load dataset------
    def get_file_names(self):
        """
        obtain file names
        returns
        -------
        file_names: list
        """
        dataset_dir = os.path.join(self.cwd, self.dataset_path)
        extention = "*.npy"
        if not os.path.exists(dataset_dir): 
            raise NotImplementedError(f"{dataset_dir} is not exist")
        #loading
        file_names = sorted(glob.glob(os.path.join(dataset_dir, extention)))

        return file_names
    
    def load_dataset(self):
        print("find %d npy files!" % len(self.file_names))
        #load input data
        for idx, file_name in tqdm(enumerate(self.file_names), desc="load dataset"):
            data = np.load(file_name, allow_pickle=True).item()
            for name in self.cfg.train.input_names:
                self.inputs[name][idx] = np_to_tensor(data[name], device=self.device)
            if self.cfg.train.condition_type == "text":
                text = data["text"]
                self.texts.append(text)
                self.labels[idx] = -1
            elif self.cfg.train.condition_type == "class":
                #folk:0 jazz:1 classical:2 other:-1 
                if data["text"] == "jazz":
                    text = "jazz"
                    label = 0
                elif data["text"] == "classical":
                    text = "classical"
                    label = 1
                else:
                    label = -1
                self.labels[idx] = label
                self.texts.append(None)
            else:
                self.labels[idx] = -1
                self.texts.append(None)
        #convert to data_loader
        if self.cfg.train.condition_type == "text":
            text_tmp = clip.tokenize(self.texts).to(self.device)
            self.data = torch.utils.data.TensorDataset(*[self.inputs[name] for name in self.cfg.train.input_names], text_tmp)
        else:
            self.data = torch.utils.data.TensorDataset(*[self.inputs[name] for name in self.cfg.train.input_names], self.labels)
        self.loader = torch.utils.data.DataLoader(self.data, batch_size=self.cfg.train.batch_size, shuffle=True)






