import torch
import numpy as np
import copy
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from einops import rearrange, reduce, repeat
from peft import get_peft_model, LoraConfig, LoraModel

from algos.Diffusion.algo import Diffusion
from algos.CLIP.algo import CLIP
from algos.VAE.algo import VAE
from algos.ProbVLM.algo import ProbVLM

class Model_Base(nn.Module):
    """
    setting each models

    params
    ------
    cfg: object
    device: torch.device #using device

    attributes
    ----------
    cfg: object
    device: torch.device
    loss_list: dict[list] #loss list
    optimizer: dict[object] #optimizer
    ema: object #Exponential Moving Average
    z_model: object #model for obtain z
    text_encoder: object #model for text encoding
    """
    def __init__(self, cfg, device):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.loss_list = dict()
        self.optimizer = dict()
        self.init_models()

    def init_models(self):
        #----set encoder----
        if self.cfg.train.encoder == "VAE":
            self.encoder = VAE(self.cfg).to(self.device)
            pretrained_model_path = self.cfg.vae.pretained_model_path
            model_name = "encoder"
        elif self.cfg.train.encoder == None:
            print("Not use encoder")
            self.encoder = None
        else:
            raise ValueError(f"Not support {self.cfg.train.encoder}")
        #set param
        if self.encoder is not None:
            self.optimizer["encoder"] = optim.Adam(self.encoder.parameters(), lr=self.cfg.train.learning_rate)
            self.loss_list["encoder"] = {"train": [], "validation": []}
            if pretrained_model_path is not None:
                self.load_params(self.encoder, pretrained_model_path, model_name, "encoder")

        #----select diffusion----
        if self.cfg.train.z_model == "Diffusion":
            self.z_model = Diffusion(cfg=self.cfg, device=self.device)
            pretrained_model_path = self.cfg.diffusion.pretained_model_path
            model_name = "diffusion"
            #params setting
            self.optimizer["z_model"] = optim.Adam(self.z_model.parameters(), lr=self.cfg.train.learning_rate)
            self.loss_list["z_model"] = {"train": [], "validation": []}
            #load pretrained model
            pretrained_lora_diffusion = False #flag of pretrained LoRA
            if pretrained_model_path is not None:
                try:
                    self.load_params(self.z_model, pretrained_model_path, model_name, "z_model")
                except Exception as e:
                    if self.cfg.diffusion.use_lora:
                        pretrained_lora_diffusion = True
                    else:
                        print(e)
                        raise ValueError(f"Can't load {pretrained_model_path}")
            #setting LoRA
            if self.cfg.diffusion.use_lora:
                print("Use LoRA")
                cfg_lora = self.cfg.diffusion.lora
                diffusion_lora_config = LoraConfig(
                    r=cfg_lora["r"],
                    lora_alpha=cfg_lora["lora_alpha"],
                    target_modules=cfg_lora["target_modules"],
                    lora_dropout=cfg_lora["lora_dropout"],
                    bias=cfg_lora["bias"],
                    inference_mode=False
                )
                #set LoRA
                self.z_model = get_peft_model(
                    self.z_model,
                    diffusion_lora_config)
                self.z_model.print_trainable_parameters()
                #set optimizer for LoRA
                lora_layers = filter(lambda p: p.requires_grad, self.z_model.parameters())
                self.optimizer["z_model"] = optim.Adam(lora_layers, lr=self.cfg.train.learning_rate)
                if pretrained_lora_diffusion:
                    print("Use pretrained LoRA")
                    self.load_params(self.z_model, pretrained_model_path, model_name, "z_model")
            #setting EMA
            if self.cfg.diffusion.use_ema:
                print("Use EMA")
                self.ema_z_model = torch.optim.swa_utils.AveragedModel(self.z_model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
        elif self.cfg.train.z_model == None:
            print("Not use z_model")
        else:
            raise ValueError(f"Not support {self.cfg.train.z_model}")
        
        #----set CLIP and ProbVLM----
        if self.cfg.train.use_clip_probvlm:
            print("Use CLIP & ProbVLM")
            self.CLIP = CLIP(self.cfg, self.device)
            self.optimizer["CLIP"] = optim.Adam(self.CLIP.model.parameters(), lr=self.cfg.train.learning_rate)
            self.loss_list["CLIP"] = {"train": [], "validation": []}
            pretrained_model_path = self.cfg.clip_probvlm.CLIP["parameter_path"]
            model_name = "CLIP"
            if pretrained_model_path is not None:
                self.load_params(self.CLIP, pretrained_model_path, model_name, "CLIP")
            #set ProbVLM
            self.ProbVLM = ProbVLM(self.cfg, self.CLIP, self.device)
            self.optimizer["ProbVLM"] = optim.Adam(self.ProbVLM.parameters(), lr=self.cfg.train.learning_rate)
            self.loss_list["ProbVLM"] = {"train": [], "validation": []}
            pretrained_model_path = self.cfg.clip_probvlm.ProbVLM["parameter_path"]
            model_name = "ProbVLM"
            pretrained_lora_probvlm = False #flag of pretrained LoRA
            if pretrained_model_path is not None:
                try:
                    self.load_params(self.ProbVLM, pretrained_model_path, model_name, "ProbVLM")
                except Exception as e:
                    if self.cfg.clip_probvlm.ProbVLM["use_lora"]:
                        pretrained_lora_probvlm = True
                    else:
                        print(e)
                        raise ValueError(f"Can't load {pretrained_model_path}")
            #setting LoRA
            if self.cfg.clip_probvlm.ProbVLM["use_lora"]:
                print("Use LoRA for ProbVLM")
                cfg_lora = self.cfg.clip_probvlm.ProbVLM["lora"]
                probvlm_lora_config = LoraConfig(
                    r=cfg_lora["r"],
                    lora_alpha=cfg_lora["lora_alpha"],
                    target_modules=cfg_lora["target_modules"],
                    lora_dropout=cfg_lora["lora_dropout"],
                    bias=cfg_lora["bias"],
                    inference_mode=False
                )
                #set LoRA
                self.ProbVLM = get_peft_model(
                    self.ProbVLM,
                    probvlm_lora_config)
                self.ProbVLM.print_trainable_parameters()
                #set optimizer for LoRA
                lora_layers = filter(lambda p: p.requires_grad, self.ProbVLM.parameters())
                self.optimizer["ProbVLM"] = optim.Adam(lora_layers, lr=self.cfg.train.learning_rate)
                if pretrained_lora_probvlm:
                    print("Use pretrained LoRA")
                    self.load_params(self.ProbVLM, pretrained_model_path, model_name, "ProbVLM")


    def load_params(self, model, model_path, model_name, model_key):
        """
        params
        ------
        model: object
        model_path: str
        model_name: str
        model_key: str
        """
        #load params
        print(f"load {model_name} from {model_path}")
        params = torch.load(model_path, map_location=self.device)
        model.load_state_dict(params["model"])
        #load optimizer
        if "optimizer" in params.keys():
            if params["optimizer"] is not None:
                self.optimizer[model_key].load_state_dict(params["optimizer"])
        #load loss
        if "loss" in params.keys():
            if params["loss"] is not None:
                self.loss_list[model_key] = params["loss"]
    
    def train_z_model(self, input, is_train=True, use_modalities=True):
        """
        params
        ------
        input: torch.tensor[modality name, ... , label(text)] #input data
        is_train: bool #train or eval
        use_modalities: bool #use modalities or not
        
        return
        ------
        loss: torch.tensor loss
        """
        if is_train:
            self.train()
            self.z_model.train()
            self.optimizer["z_model"].zero_grad()
        else:
            self.eval()
            self.z_model.eval()
        loss = 0
        #入力データの整形
        if len(input) == 1:
            x = input[0]
        else:
            x = input[:-1]
        if use_modalities:
            modalities = len(x) #num of modalities
            if modalities == 1:
                x = x[0].to(self.device)
            else:
                x = torch.cat(x, dim=1).to(self.device) #(b, c, t, p)
        else:
            x = x.to(self.device)
        x = x.permute(0, 1, 3, 2) #(b, c, p, t)
        #split data
        num_sample = x.shape[-1]//x.shape[-2]
        if num_sample > 1:
            x = torch.chunk(x, num_sample, dim=-1)
            x = torch.concat(x, dim=0)
        #condition
        cond = None
        if self.cfg.train.condition_type == "text":
            #text input woth probability
            if np.random.rand() < self.cfg.diffusion.p_cond:
                text = input[-1].to(self.device)
                cond= self.CLIP.model.encode_text(text).to(torch.float32)
        elif self.cfg.train.condition_type == "class":
            cond = input[-1].to(self.device)
        #Encode
        if self.cfg.train.encoder is not None:
            x = self.encoder.get_z(x)
        #統合
        if num_sample > 1:
            x = torch.chunk(x, num_sample, dim=0)
            x = torch.concat(x, dim=-1)
        x = x.permute(0, 1, 3, 2) #(b, c, t, p)
        #Diffusion
        if is_train:
            loss = self.z_model(x, is_train=is_train, cond=cond)
        else:
            loss = self.z_model.forward_val(x, is_train=is_train, cond=cond)

        # Update model parameters
        if is_train:
            loss.backward()
            self.optimizer["z_model"].step()
            if self.cfg.diffusion.use_ema:
                self.ema_z_model.update_parameters(self.z_model)


        return loss.item()
    
    def train_encoder(self, input, is_train=True, use_modalities=True):
        """
        params
        ------
        input: torch.tensor[modality name, ... , label(text)]
        is_train: bool  #train or eval 
        use_modalities: bool  #use modalities or not

        return
        ------
        loss: torch.tensor loss
        """
        if is_train:
            self.train()
            self.encoder.train()
            self.optimizer["encoder"].zero_grad()
        else:
            self.eval()
            self.encoder.eval()
        loss = 0
        #入力データの整形
        if len(input) == 1:
            x = input[0]
        else:
            x = input[:-1]
        if use_modalities:
            modalities = len(x) #num of modalities
            if modalities == 1:
                x = x[0].to(self.device)
            else:
                x = torch.cat(x, dim=1).to(self.device)
        else:
            x = x.to(self.device)
        x = x.permute(0, 1, 3, 2) #(b, c, p, t)
        #split data
        num_sample = x.shape[-1]//x.shape[-2]
        if num_sample > 1:
            x = torch.chunk(x, num_sample, dim=-1)
            x = torch.concat(x, dim=0)
        if is_train:
            loss = self.encoder(x)
            loss.backward()
            self.optimizer["encoder"].step()
        else:
            loss = self.encoder.forward_val(x)
        
        return loss.item()
    
    def train_CLIP(self, input, is_train=True):
        """
        params
        ------
        input: torch.tensor[image_conv, text_conv]
        is_train: bool

        return
        ------
        loss: torch.tensor loss
        """
        loss = 0
        if is_train:
            self.train()
            self.CLIP.model.train()
            loss = self.CLIP.train(input)
        else:
            self.eval()
            self.CLIP.model.eval()
            with torch.no_grad():
                loss = self.CLIP.train(input, is_train=False)
        
        return loss.item()
    
    def train_ProbVLM(self, input, is_train=True):
        """
        params
        ------
        input: torch.tensor[image_conv, text_conv]
        is_train: bool

        return
        ------
        loss: torch.tensor loss
        """
        loss = 0
        img = input[0]
        text = input[1]
        if is_train:
            self.train()
            self.ProbVLM.train()
            self.optimizer["ProbVLM"].zero_grad()
            loss = self.ProbVLM(img, text)
            loss.backward()
            self.optimizer["ProbVLM"].step()
        else:
            self.eval()
            self.ProbVLM.eval()
            with torch.no_grad():
                loss = self.ProbVLM(img, text)
        
        return loss.item()

    def get_z(self, data):
        """
        data: torch.utils.data.DataLoader
        """
        for i, batch in enumerate(data):
            z = self.z_model.get_z(batch[0].to(self.device))
            label = batch[-1]
            break
        #z_compresion
        if self.cfg.train.z_compresion:
            z = self.compresion_z.encode(z).to("cpu")
        return z.detach().numpy(), label.detach().numpy()
    
    def get_mu_sigma_vae(self, x):
        """
        x: torch.utils.data.DataLoader #input data
        """
        sample, mu, sigma = self.encoder.get_mu_sigma(x.to(self.device))
        return sample.to("cpu").detach().numpy(), mu.to("cpu").detach().numpy(), sigma.to("cpu").detach().numpy()
    
    def sampling_diffusion(self, cond=None):
        """
        sammpling from diffusion model
        cond: str #condition text or label

        return
        ------
        output: dict
        """
        #set parameters
        cond_orig = cond
        #ノイズの生成
        sampler = self.z_model.sampler
        #生成するサンプル数
        noise = torch.randn([self.cfg.diffusion.sampling_size, 
                            *self.cfg.diffusion.sampling_shape]).to(self.device)
        
        #condtion encoding
        if cond is not None and self.cfg.train.condition_type == "text":
            assert len(cond) == self.cfg.diffusion.sampling_size
            cond = self.CLIP.text_encoding(cond)
            cond = cond.reshape(self.cfg.diffusion.sampling_size, -1).to(torch.float32)

        #sampling
        sampled_h = sampler(noise, cond=cond, decoder=self.encoder)
        sampled_h = sampled_h.permute(0, 1, 3, 2) #(b, c, p, t)
        if sampled_h.shape[-2] > sampled_h.shape[-1]:
            sampled_h = sampled_h.permute(0, 1, 3, 2)
        num_sample = sampled_h.shape[-1]//sampled_h.shape[-2]

        #decode
        if self.cfg.train.encoder is not None:
            sumple_size = len(sampled_h)
            sumple_chunk = sumple_size//50
            if sumple_size>50:
                sampled_h_all = torch.chunk(sampled_h, sumple_chunk, dim=0)
            else:
                sampled_h_all = [sampled_h]
            out_sampled_h = []
            for sampled_h in sampled_h_all:
                sampled_h = sampled_h.to(self.device)
                if num_sample == 1:
                    sampled_h = self.encoder.decode_z(sampled_h, scale_factor=True)
                else:
                    sampled_h /= self.cfg.vae.scale_factor
                    sampled_h = torch.chunk(sampled_h, num_sample, dim=-1)
                    sampled_h = torch.concat(sampled_h, dim=0)
                    sampled_h = self.encoder.decode_z(sampled_h, scale_factor=False)
                    sampled_h = torch.concat(torch.chunk(sampled_h, num_sample, dim=0), dim=-1)
                out_sampled_h.append(sampled_h)
            sampled_h = torch.concat(out_sampled_h, dim=0)
        sampled_h[sampled_h <= -0.95] = -1.
        sampled_h = sampled_h.permute(0, 1, 3, 2)
        output = dict()
        for i, key in enumerate(self.cfg.train.input_names):
            channel = self.cfg.train.input_shapes[key][0]
            hidden_tmp = sampled_h[:, channel*i:channel*(i+1), :, :]
            output[key] = hidden_tmp.to("cpu")
        if not self.cfg.train.condition_type == None:
            output["text"] = cond_orig
        else:
            output["text"] = None

        return output
    
    def sampling_vae(self, test_data):
        """
        sampling from VAE model

        params
        ------
        test_data: dict

        return
        ------
        output: dict[torch.tensor]
        """
        input = []
        for key in self.cfg.train.input_names:
            input.append(test_data[key])
        if len(input) == 1:
            input = input[0].to(self.device)
        else:
            input = torch.cat(input, dim=1).to(self.device)
        #sampling
        input = input.permute(0, 1, 3, 2)
        rec = self.encoder.sampling(input)
        rec = rec.permute(0, 1, 3, 2)
        rec[rec <= -0.95] = -1.
        output = dict()
        for i, key in enumerate(self.cfg.train.input_names):
            channel = self.cfg.train.input_shapes[key][0]
            hidden_tmp = rec[:, channel*i:channel*(i+1), :, :]
            output[key] = hidden_tmp.to("cpu")
        
        return output

        