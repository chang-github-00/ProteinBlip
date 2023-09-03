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

@registry.register_model("mini_gpt4_adapter_generation")
class MiniGPT4_Adapter_Generation(Blip2Base):
    """
    BLIP2 GPT-LLAMA model. 
    
    Change Q-former to perceiver-based adapter.
    
    """
    
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4_generation.yaml",
    }
    
    def __init__(
        self,
        esm_model="esm2_t33_650M_UR50D",
        generator="",
        freeze_generator=True,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        esm_precision="fp16",
        freeze_esm=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        max_generation_len=500,
        max_prompt_len=200,
        end_sym='\n',
        adapter_depth=2,
        adapter_dim_head=64,
        adapter_heads=8,
        adapter_num_latent_tokens=32,
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        need_encoder=False,
        need_decoder=False,
    ):
        
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        
        print('start loading LLAMA')
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
        
        
        if need_encoder:
            print('Loading ESM')
            self.esm_encoder, self.esm_alphabet = self.init_protein_encoder(esm_model, esm_precision)
            
            print('Loading ESM Done')
            
            if freeze_esm:
                for name, param in self.esm_encoder.named_parameters():
                    param.requires_grad = False
                self.esm_encoder = self.esm_encoder.eval()
                
                logging.info("freeze esm encoder")
            
            
            self.protein_adapter = PerceiverAdapter(self.esm_encoder.embed_dim, 
                                                    depth = adapter_depth,
                                                    dim_head = adapter_dim_head,
                                                    heads = adapter_heads,
                                                    num_latents = adapter_num_latent_tokens)
            self.esm_llama_proj = nn.Linear(
                self.esm_encoder.embed_dim, self.llama_model.config.hidden_size
            )
            
        
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if need_decoder:
            print('Loading Generator')
            
            self.generator = AutoModelForCausalLM.from_pretrained(generator)
            self.generator_tokenizer = MixgptTokenizer.from_pretrained(generator)
            self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
            
            if freeze_generator:
                for name, param in self.generator.named_parameters():
                    param.requires_grad = False
                
                logging.info("freeze generator")
                
            self.output_adapter = PerceiverAdapter(self.llama_model.config.hidden_size,
                                                depth = adapter_depth,
                                                dim_head = adapter_dim_head,
                                                heads = adapter_heads,
                                                num_latents = adapter_num_latent_tokens)
            self.llama_generator_proj = nn.Linear(
                self.llama_model.config.hidden_size, self.generator.config.hidden_size
            )
        
        self.max_txt_len = max_txt_len
        self.max_generation_len = max_generation_len
        self.max_prompt_len = max_prompt_len
        self.end_sym = end_sym
        self.prompt_template = prompt_template
        
        
        
        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<proteinHere>" in raw_prompt]
            self.raw_prompt_list = filted_prompts
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
        
        protein_embeds = self.protein_adapter(protein_embeds)
        # input llama is of shape [B, 32, 5120]
        inputs_llama = self.esm_llama_proj(protein_embeds)
        # atts_llama is of shape [B, 32]
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
        return inputs_llama, atts_llama
    
    def encode_multiple_protein(self, protein_sequences_dict, batch_size=1):
        inputs_llama_dict = {}
        atts_llama_dict = {}
        for id in protein_sequences_dict:
            protein_sequences = protein_sequences_dict[id]
            inputs_llama, atts_llama = self.encode_protein(protein_sequences, batch_size)
            inputs_llama_dict[id] = inputs_llama
            atts_llama_dict[id] = atts_llama
        return inputs_llama_dict, atts_llama_dict
    
    def prompt_wrap(self, protein_embeds, atts_protein, prompt): # single prompt for a batch
        if prompt:
            batch_size = protein_embeds.shape[0]
            p_before, p_after = prompt.split('<proteinHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(protein_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(protein_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_protein_embeds = torch.cat([p_before_embeds, protein_embeds, p_after_embeds], dim=1)
            wrapped_atts_protein = atts_protein[:, :1].expand(-1, wrapped_protein_embeds.shape[1])
            return wrapped_protein_embeds, wrapped_atts_protein
        else:
            return protein_embeds, atts_protein
        
    def prompt_list_wrap(self, protein_embeds, atts_protein, prompt_list): # multiple prompts for a batch
        if prompt_list:
            prompt_list = [p if "<proteinHere>" in p else "<protein><proteinHere></protein>" for p in prompt_list]   # filter prompts without <proteinHere>
            prompt_list = [self.prompt_template.format(p) for p in prompt_list] # add formatting, like ###Human: {} ###Assistant: 
            
            batch_size = protein_embeds.shape[0]
            prompt_list_no_ph = ["".join(prompt.split('<proteinHere>')) for prompt in prompt_list]
            p_before = [sentence.split("<proteinHere>")[0] for sentence in prompt_list]  
            
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False, padding="longest",max_length=self.max_prompt_len, truncation=True).to(protein_embeds.device)
            
            p_before_lengths = p_before_tokens.attention_mask.sum(dim=1)  
            
            p_all_tokens = self.llama_tokenizer(  
                prompt_list_no_ph, return_tensors="pt", add_special_tokens=False, padding="longest", max_length=self.max_prompt_len, truncation=True).to(protein_embeds.device)
            
            
            p_all_embeds = self.llama_model.model.embed_tokens(p_all_tokens.input_ids)
            
            wrapped_protein_embeds = []
            wrapped_atts_protein = []
            for i, length in enumerate(p_before_lengths):
                wrapped_protein_embeds.append(torch.cat([p_all_embeds[i, :length], protein_embeds[i], p_all_embeds[i, length:]], dim=0))
                
            wrapped_protein_embeds = torch.stack(wrapped_protein_embeds, dim=0)
            wrapped_atts_protein = torch.cat([atts_protein, p_all_tokens.attention_mask], dim=1)
            return wrapped_protein_embeds, wrapped_atts_protein
        else:
            return protein_embeds, atts_protein
    
    def prompt_wrap_multiple(self, protein_embeds, atts_protein, prompt): 
        raise NotImplementedError
    
    def prompt_list_wrap_multiple(self, protein_embeds_dict, atts_protein_dict, prompt_list, symbol_id=29930): # multiple prompts for a batch
        device = protein_embeds_dict[1].device
        wrapped_protein_embeds = []
        wrapped_atts_protein = []
        if prompt_list:
            prompt_list = [self.prompt_template.format(p) for p in prompt_list] # add formatting, like ###Human: {} ###Assistant: 
            
            pattern = r'<proteinHere(?:_\d+)?>'
            pattern_num = r'<proteinHere_(\d+)>'
            symbol = self.llama_tokenizer.convert_ids_to_tokens(symbol_id)
            
            for i, prompt in enumerate(prompt_list):
                prompt_no_ph = re.sub(pattern, symbol, prompt)
                protein_ids = [int(p) for p in re.findall(pattern_num, prompt)]# 0 for <proteinHere>, 1 for <proteinHere_1>, 2 for <proteinHere_2>...
                protein_embeds = [protein_embeds_dict[id][i] for id in protein_ids] # list of protein embeds to insert into placeholders
                
                prompt_tokens = self.llama_tokenizer(prompt_no_ph, return_tensors="pt", add_special_tokens=False, max_length=self.max_prompt_len, truncation=True).to(device)
                
                prompt_embeds = self.llama_model.model.embed_tokens(prompt_tokens.input_ids)
                
                # to do: add protein embeds to prompt_embeds
                
                symbol_indices = (prompt_tokens.input_ids[0] == symbol_id).nonzero().squeeze()
                
                for j, index in enumerate(symbol_indices):
                    index += protein_embeds[j].shape[0] * j  #adjust index for previous insertions
                    prompt_embeds = torch.cat([prompt_embeds[:, :index], protein_embeds[j].unsqueeze(0), prompt_embeds[:, index+1:]], dim=1)
                
                atts_protein = prompt_tokens.attention_mask[:, len(symbol_indices):] # remove the first token, which is <proteinHere>
                for atts in atts_protein_dict:
                    atts_protein = torch.cat([atts_protein_dict[atts][i].unsqueeze(0), atts_protein], dim=1) 
                
                wrapped_protein_embeds.append(prompt_embeds)
                wrapped_atts_protein.append(atts_protein)
            
            # align the length of wrapped_protein_embeds and wrapped_atts_protein
            
            symbol_embeds = self.llama_model.model.embed_tokens(torch.tensor(symbol_id).unsqueeze(0).to(device))
            max_length = max([embeds.shape[1] for embeds in wrapped_protein_embeds])
            
            for i, embeds in enumerate(wrapped_protein_embeds):
                wrapped_protein_embeds[i] = torch.cat([embeds, symbol_embeds.repeat(1, max_length-embeds.shape[1], 1)], dim=1).squeeze(0)
                wrapped_atts_protein[i] = torch.cat([wrapped_atts_protein[i], torch.zeros(1, max_length-embeds.shape[1]).to(device)], dim=1).squeeze(0)
            
            wrapped_protein_embeds = torch.stack(wrapped_protein_embeds, dim=0)
            wrapped_atts_protein = torch.stack(wrapped_atts_protein, dim=0)
            
            return wrapped_protein_embeds, wrapped_atts_protein
        else:
            return protein_embeds, atts_protein
    
    def forward(self, samples):
        mode = samples["mode"][0] if "mode" in samples else None
        if mode == "text2protein":
            return self.forward_text_to_protein(samples)
        elif mode == "protein+text2protein":
            return self.forward_protein_text_to_protein(samples)
        else:
            return self.forward_protein_to_text(samples)
    
    
    def forward_protein_text_to_protein(self, samples):
        device = self.llama_model.device
        
        chains = [key for key in samples.keys() if key.startswith('chain')]
        target_chains = [key for key in samples.keys() if key.startswith('target_chain')]
        
        if len(chains) == 1: # single protein mode
            protein_sequences = samples["chain"]
            target_chains = samples["target_chain"]
            protein_embeds, atts_protein = self.encode_protein(protein_sequences)
            
            if hasattr(samples, 'instruction_split') or 'instruction_split' in samples:  # Instruction tuning mode        # remember to add this attribute to the dataset
                # print('Instruction tuning mode')
                pqa_prompt = samples["prompt"]
                protein_embeds, atts_protein = self.prompt_list_wrap(protein_embeds, atts_protein, pqa_prompt)
            elif self.prompt_list:
                prompt = random.choice(self.prompt_list)
                protein_embeds, atts_protein = self.prompt_wrap(protein_embeds, atts_protein, prompt)
        
        else: # multiple protein mode
            protein_sequences_dict = dict() # a dict of protein sequences, each key is a chain id, each value is a list of protein sequences
            for attr in samples:
                if attr.startswith('chain'):
                    if attr == 'chain':
                        id = 0
                    else:
                        id = int(attr.split('_')[1])
                    protein_sequences_dict[id] = samples[attr]
                    
            protein_embeds_dict, atts_protein_dict = self.encode_multiple_protein(protein_sequences_dict)
        
            if hasattr(samples, 'instruction_split') or 'instruction_split' in samples:  # Instruction tuning mode        # remember to add this attribute to the dataset
                # print('Instruction tuning mode')
                pqa_prompt = samples["prompt"]
                protein_embeds, atts_protein = self.prompt_list_wrap_multiple(protein_embeds_dict, atts_protein_dict, pqa_prompt)
            elif self.prompt_list:
                prompt = random.choice(self.prompt_list)
                protein_embeds, atts_protein = self.prompt_wrap_multiple(protein_embeds_dict, atts_protein_dict, prompt)

        # text to be encoded 
        # text = [t + self.end_sym for t in samples["text_input"]]
        
        # # encode text
        # text_input = self.llama_tokenizer(text, 
        #                                   return_tensors="pt", 
        #                                   add_special_tokens=False, 
        #                                   padding="longest", 
        #                                   max_length=self.max_text_len, 
        #                                   truncation=True).to(self.llama_model.device)
        
        batch_size = protein_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=torch.int64,
                         device=device) * self.llama_tokenizer.bos_token_id

        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_protein[:, :1]

        # text_embeds = self.llama_model.model.embed_tokens(text_input.input_ids)

        inputs_embeds = torch.cat([bos_embeds, protein_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_protein], dim=1)
        
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        
        hidden_states = outputs.hidden_states[-1] # last layer hidden states
        generation_query = self.output_adapter(hidden_states) # query for generation
        generation_query = self.llama_generator_proj(generation_query) # project to the size of vocab
        attns_query = torch.ones(generation_query.size()[:-1], dtype=torch.long).to(device)
        
        to_regress_tokens = self.generator_tokenizer(target_chains,
                                                     return_tensors="pt",
                                                     padding="longest",
                                                     truncation=True,
                                                     max_length=self.max_generation_len,
                                                     add_special_tokens=False).to(device) 
        
        
        # add eos token to to_regress_tokens
        
        eos_token_id = self.generator_tokenizer.eos_token_id
        to_regress_tokens.input_ids = torch.cat([to_regress_tokens.input_ids, torch.ones([to_regress_tokens.input_ids.shape[0], 1], dtype=torch.long).to(device) * eos_token_id], dim=1)
        to_regress_tokens.attention_mask = torch.cat([to_regress_tokens.attention_mask, torch.ones([to_regress_tokens.attention_mask.shape[0], 1], dtype=torch.long).to(device)], dim=1)
        
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.generator_tokenizer.pad_token_id, -100
        )                                   
        
        empty_targets = (
            torch.ones([attns_query.shape[0], attns_query.shape[1]+1],
                       dtype=torch.long).to(device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        batch_size = generation_query.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.generator_tokenizer.bos_token_id
        bos_embeds = self.generator.biogpt.embed_tokens(bos)
        atts_bos = attns_query[:, :1]
        
        to_regress_embeds = self.generator.biogpt.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, generation_query, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, attns_query, to_regress_tokens.attention_mask], dim=1)
        
        with self.maybe_autocast():
            outputs = self.generator(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss
        return {"loss": loss}
    
    def forward_text_to_protein(self, samples):
        # only support single protein mode
        device = self.llama_model.device
        protein_sequences = samples["chain"] # protein sequence to be generated
        protein_sequences = ["[P]" + sequence  for sequence in protein_sequences] # add [P] to the beginning of the sequence
        
        if hasattr(samples, 'instruction_split') or 'instruction_split' in samples:  # Instruction tuning mode 
            pqa_prompt = samples["prompt"]
        elif self.prompt_list:
            pqa_prompt = random.choice(self.prompt_list)
        
        text_input = samples["text_input"]
        text = [prompt + t + self.end_sym for prompt, t in zip(pqa_prompt, text_input)] # text to be encoded
        tokenized_text = self.llama_tokenizer(text, 
                                              return_tensors="pt", 
                                              add_special_tokens=False, 
                                              padding="longest", 
                                              max_length=self.max_txt_len, 
                                              truncation=True).to(device)
        
        with self.maybe_autocast():
            outputs = self.llama_model(
                **tokenized_text,
                output_hidden_states=True,
            )
        
        hidden_states = outputs.hidden_states[-1] # last layer hidden states
        generation_query = self.output_adapter(hidden_states) # query for generation
        generation_query = self.llama_generator_proj(generation_query) # project to the size of vocab
        attns_query = torch.ones(generation_query.size()[:-1], dtype=torch.long).to(device)
        
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
        
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.generator_tokenizer.pad_token_id, -100
        )                                   
        
        empty_targets = (
            torch.ones([attns_query.shape[0], attns_query.shape[1]+1],
                       dtype=torch.long).to(device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        batch_size = generation_query.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.generator_tokenizer.bos_token_id
        bos_embeds = self.generator.biogpt.embed_tokens(bos)
        atts_bos = attns_query[:, :1]
        
        to_regress_embeds = self.generator.biogpt.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, generation_query, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, attns_query, to_regress_tokens.attention_mask], dim=1)
        
        with self.maybe_autocast():
            outputs = self.generator(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss
        return {"loss": loss}

        
    def forward_protein_to_text(self, samples):
        chains = [key for key in samples.keys() if key.startswith('chain')]
        if len(chains) == 1: # single protein mode
            protein_sequences = samples["chain"]
            protein_embeds, atts_protein = self.encode_protein(protein_sequences)
            
            if hasattr(samples, 'instruction_split') or 'instruction_split' in samples:  # Instruction tuning mode        # remember to add this attribute to the dataset
                # print('Instruction tuning mode')
                pqa_prompt = samples["prompt"]
                protein_embeds, atts_protein = self.prompt_list_wrap(protein_embeds, atts_protein, pqa_prompt)
            elif self.prompt_list:
                prompt = random.choice(self.prompt_list)
                protein_embeds, atts_protein = self.prompt_wrap(protein_embeds, atts_protein, prompt)
        
        else: # multiple protein mode
            protein_sequences_dict = dict() # a dict of protein sequences, each key is a chain id, each value is a list of protein sequences
            for attr in samples:
                if attr.startswith('chain'):
                    if attr == 'chain':
                        id = 0
                    else:
                        id = int(attr.split('_')[1])
                    protein_sequences_dict[id] = samples[attr]
                    
            protein_embeds_dict, atts_protein_dict = self.encode_multiple_protein(protein_sequences_dict)
        
            if hasattr(samples, 'instruction_split') or 'instruction_split' in samples:  # Instruction tuning mode        # remember to add this attribute to the dataset
                # print('Instruction tuning mode')
                pqa_prompt = samples["prompt"]
                protein_embeds, atts_protein = self.prompt_list_wrap_multiple(protein_embeds_dict, atts_protein_dict, pqa_prompt)
            elif self.prompt_list:
                prompt = random.choice(self.prompt_list)
                protein_embeds, atts_protein = self.prompt_wrap_multiple(protein_embeds_dict, atts_protein_dict, prompt)

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
        generator = cfg.get("generator")
        
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
        max_prompt_len = cfg.get("max_prompt_len", 200)
        end_sym = cfg.get("end_sym", '\n')

        # adapter_depth=2,
        # adapter_dim_head=64,
        # adapter_heads=8,
        # adapter_num_latent_tokens=32,
        
        
        adapter_depth = cfg.get("adapter_depth", 2)
        adapter_dim_head = cfg.get("adapter_dim_head", 64)
        adapter_heads = cfg.get("adapter_heads", 8)
        adapter_num_latent_tokens = cfg.get("adapter_num_latent_tokens", 32)
        
        need_encoder = cfg.get("need_encoder", False)
        need_decoder = cfg.get("need_decoder", False)
    
        
        model = cls(
            esm_model=esm_model,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            esm_precision=esm_precision,
            freeze_esm=freeze_esm,
            num_query_token=num_query_token,
            llama_model=llama_model,
            generator=generator,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            max_prompt_len=max_prompt_len,
            end_sym=end_sym,
            low_resource=low_resource,  # use 8 bit and put vit in cpu
            device_8bit=device_8bit,  # the device of 8bit model should be set when loading and cannot be changed anymore.
            adapter_depth=adapter_depth,
            adapter_dim_head=adapter_dim_head,
            adapter_heads=adapter_heads,
            adapter_num_latent_tokens=adapter_num_latent_tokens,
            need_encoder=need_encoder,
            need_decoder=need_decoder
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model

    
    def encode_text(self, text):
        device = self.llama_model.device
        tokenized_text = self.llama_tokenizer(text, 
                                          return_tensors="pt", 
                                          add_special_tokens=False, 
                                          padding="longest", 
                                          max_length=self.max_txt_len, 
                                          truncation=True).to(device)
        with self.maybe_autocast():
            outputs = self.llama_model(
                **tokenized_text,
                output_hidden_states=True,
            )
        
            hidden_states = outputs.hidden_states[-1] # last layer hidden states
            generation_query = self.output_adapter(hidden_states) # query for generation
            generation_query = self.llama_generator_proj(generation_query) # project to the size of vocab
            attns_query = torch.ones(generation_query.size()[:-1], dtype=torch.long).to(device)
            
        return generation_query, attns_query