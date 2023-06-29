
import argparse
import time
from PIL import Image
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

import dataclasses
from typing import List, Tuple, Any

from minigpt4.common.registry import registry

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

class ProteinTextGenerator:
    def __init__(self, model, vis_processor, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        self.prompt_template = self.model.prompt_template
        self.prompt_list = self.model.raw_prompt_list
        
    def get_context_emb(self, queries, protein_list): # Default format : protein + query    => bos + prompt <protein> + query
        protein_embs = []
        for protein in protein_list:
            protein_emb, _ = self.model.encode_protein([protein]) # should input a list of protein, output shape [1, 32, 4096]
            protein_embs.append(protein_emb)
            
        # wrap prompt around query: 
        prompt = random.choice(self.prompt_list)
        queries = [self.prompt_template.format(prompt + query) for query in queries]
        p_before = [query.split('<proteinHere>')[0] for query in queries]
        p_after = [query.split('<proteinHere>')[1] for query in queries]
        
        p_before_tokens = self.model.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(self.device).input_ids
        p_after_tokens = self.model.llama_tokenizer(p_after, return_tensors="pt", add_special_tokens=True).to(self.device).input_ids
        
        
        p_before_embs = [self.model.llama_model.model.embed_tokens(p_before_t).unsqueeze(0) for p_before_t in p_before_tokens]
        p_after_embs = [self.model.llama_model.model.embed_tokens(p_after_t).unsqueeze(0) for p_after_t in p_after_tokens]
        
        mixed_embs = [ torch.cat([query_after[:,:1], query_before, protein, query_after[:,1:]], dim=1)  for query_before, protein, query_after in zip(p_before_embs, protein_embs, p_after_embs)] # bos + protein + query 
        mixed_embs = torch.cat(mixed_embs, dim=0)
        return mixed_embs
    

    def generate(self, queries, protein_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        
        embs = self.get_context_emb(queries, protein_list)
        
        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        # output_text = output_text.split('###')[0]  # remove the stop sign '###'
        # output_text = output_text.split('Assistant:')[-1].strip()
        # conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()