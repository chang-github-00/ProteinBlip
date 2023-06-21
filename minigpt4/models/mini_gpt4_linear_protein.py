import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import sys

from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
from minigpt4.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer
import esm

@registry.register_model("mini_gpt4_linear")
class MiniGPT4_Linear(Blip2Base):
    """
    BLIP2 GPT-LLAMA model. 
    
    Change Q-former to projection
    
    """
    
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4.yaml",
    }
    
    def __init__(
        self,
        esm_model="esm2_t33_650M_UR50D",
        drop_path_rate=0,
        use_grad_checkpoint=False,
        esm_precision="fp16",
        freeze_esm=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        
        print('Loading ESM')
        self.esm_encoder, self.esm_alphabet = self.init_protein_encoder(esm_model, esm_precision)
        
        print('Loading ESM Done')
        
        if freeze_esm:
            for name, param in self.esm_encoder.named_parameters():
                param.requires_grad = False
            self.esm_encoder = self.esm_encoder.eval()
            
            logging.info("freeze esm encoder")
            
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')
        
        self.esm_llama_proj = nn.Linear(
            self.esm_encoder.embed_dim, self.llama_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        
        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<proteinHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []
        
    @classmethod
    def init_protein_encoder(cls, model_name, precision="fp32"): # to do: add support for fp16
        if model_name == "esm2_t33_650M_UR50D":
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        
        return model, alphabet
        
    def encode_protein(self, protein_sequences, batch_size=1):
        # device = protein_encode.device
        # # with self.maybe_autocast():
        # protein_embeds = protein_encode.to(device)
        device = next(self.esm_encoder.parameters()).device  
        batch_converter = self.esm_alphabet.get_batch_converter()
        sequences = [(f"protein{i}", seq) for i, seq in enumerate(protein_sequences)]
        batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
        if batch_tokens.shape[1] > 1024:
            batch_tokens = batch_tokens[:, :1024]
        
        batches = []   # because esm is a fairly large model, we split the batch into smaller batches so that it fits into the gpu
        for i in range(0, batch_tokens.shape[0], batch_size):  
            batch = batch_tokens[i:i + batch_size]  
            batches.append(batch)  

        protein_embeds = []
        for batch in batches:
            with torch.no_grad():
                batch = batch.to(device)
                results = self.esm_encoder(batch, repr_layers=[33], return_contacts=False)
                token_embeddings = results["representations"][33]
                token_embeddings = token_embeddings[:, 1:-1, :]
            protein_embeds.append(token_embeddings)
        protein_embeds = torch.cat(protein_embeds, dim=0)
        
        # input llama is of shape [B, 32, 5120]
        inputs_llama = self.esm_llama_proj(protein_embeds)
        # atts_llama is of shape [B, 32]
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
        return inputs_llama, atts_llama
    
    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split('<proteinHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img
        
    
    def forward(self, samples):
        protein_sequences = samples["chain"]  # a list of protein sequences ? 
        protein_embeds, atts_protein = self.encode_protein(protein_sequences)
        
        if hasattr(samples, 'instruction_split'):  # Instruction tuning mode        # remember to add this attribute to the dataset
            print('Instruction tuning mode')
            pqa_prompt = '###Human: <protein><proteinHere></protein> '
            protein_embeds, atts_protein = self.prompt_wrap(protein_embeds, atts_protein, pqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)

        self.llama_tokenizer.padding_side = "right"
        
        text = [t + self.end_sym for t in samples["text_input"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(protein_embeds.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_protein.shape[0], atts_protein.shape[1]+1],
                       dtype=torch.long).to(protein_embeds.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = protein_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id

        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_protein[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)

        inputs_embeds = torch.cat([bos_embeds, protein_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_protein, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
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
        
        
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        esm_model = cfg.get("esm_model", "esm2_t33_650M_UR50D")
        esm_precision = cfg.get("esm_precision", "fp32")
        freeze_esm = cfg.get("freeze_esm", True)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        # vit_precision = cfg.get("vit_precision", "fp16")
        # freeze_protein_encoder = cfg.get("freeze_protein_encoder", True)
        # freeze_qformer = cfg.get("freeze_qformer", True)
        
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)



        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        
        model = cls(
            esm_model=esm_model,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            esm_precision=esm_precision,
            freeze_esm=freeze_esm,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,  # use 8 bit and put vit in cpu
            device_8bit=device_8bit,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
