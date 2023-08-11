import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import sys
import re

from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
from minigpt4.models.modeling_llama import LlamaForCausalLM
from minigpt4.models.helpers import PerceiverAdapter
from transformers import LlamaTokenizer
from  transformers import AutoTokenizer, AutoModelForCausalLM
import esm
from minigpt4.models.molxpt_tokenizer import MixgptTokenizer

@registry.register_model("mixpt")
class Mixpt(Blip2Base):
    """
    BLIP2 GPT-LLAMA model. 
    
    Change Q-former to perceiver-based adapter.
    
    """
    
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4_generation.yaml",
    }
    
    def __init__(
        self,
        generator="",
        freeze_generator=True,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        max_generation_len=500,
    ):
        
        super().__init__()

        print('Loading Generator')
        
        self.generator = AutoModelForCausalLM.from_pretrained(generator)
        self.generator_tokenizer = MixgptTokenizer.from_pretrained(generator)
        self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
        
        if freeze_generator:
            for name, param in self.generator.named_parameters():
                param.requires_grad = False
            
            logging.info("freeze generator")
    
        self.max_generation_len = max_generation_len

    def forward(self, samples):
        mode = samples["mode"][0] if "mode" in samples else None
        if mode == "text2protein":
            return self.forward_text_to_protein(samples)
        elif mode == "protein+text2protein":
            return self.forward_protein_text_to_protein(samples)
        else:
            return self.forward_protein_to_text(samples)
    
    def forward_text_to_protein(self, samples):
        # only support single protein mode
        device = self.generator.device
        protein_sequences = samples["chain"] # protein sequence to be generated
        protein_sequences = ["[P]" + sequence  for sequence in protein_sequences] # add [P] to the beginning of the sequence
        
        to_regress_tokens = self.generator_tokenizer(protein_sequences,
                                                     return_tensors="pt",
                                                     padding="longest",
                                                     truncation=True,
                                                     max_length=self.max_generation_len,
                                                     add_special_tokens=False).to(device)      
        
        
        # add eos token to to_regress_tokens
        
        eos_token_id = self.generator_tokenizer.eos_token_id
        to_regress_tokens.input_ids = torch.cat([to_regress_tokens.input_ids, torch.ones([to_regress_tokens.input_ids.shape[0], 1], dtype=torch.long).to(device) * eos_token_id], dim=1)
        to_regress_tokens.attention_mask = torch.cat([to_regress_tokens.attention_mask, torch.ones([to_regress_tokens.attention_mask.shape[0], 1], dtype=torch.long).to(device)], dim=1)
    
        batch_size = to_regress_tokens.input_ids.shape[0]
        
        to_regress_embeds = self.generator.biogpt.embed_tokens(to_regress_tokens.input_ids)
        
        
        
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.generator_tokenizer.pad_token_id, -100
        )                                   
        
        special_start_id = self.generator_tokenizer.encoder['[P]</w>']
        
        targets = targets.masked_fill(
            targets == special_start_id, -100
        )
        
        
        with self.maybe_autocast():
            outputs = self.generator(
                inputs_embeds=to_regress_embeds,
                attention_mask=to_regress_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss
        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):

        # vit_model = cfg.get("vit_model", "eva_clip_g")
        # q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        # img_size = cfg.get("image_size")
        
        generator = cfg.get("generator")
    
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        max_generation_len = cfg.get("max_generation_len", 200)
        
        freeze_generator = cfg.get("freeze_generator", True)
        
        
        
        model = cls(
            generator=generator,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            max_generation_len=max_generation_len,
            freeze_generator=freeze_generator,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
