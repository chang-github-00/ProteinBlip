import argparse
import os
import random
from tqdm import tqdm
import numpy as np
import torch
import json
import torch.backends.cudnn as cudnn
from rouge import Rouge

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from minigpt4.conversation.simple_generator import ProteinTextGenerator

import numpy as np
from goatools.obo_parser import GODag  
import re
import random
from tqdm import tqdm
import json
  
def parse_go_terms(file_path):    
    go_terms = {}    
    go_names = {}  
    with open(file_path, 'r') as file:    
        current_go_id = None    
        for line in file:    
            # Check for a new term    
            if line.startswith('[Term]'):    
                current_go_id = None    
            # Check for an ID    
            elif line.startswith('id:'):    
                current_go_id = line.strip().split(' ')[1]    
            # Check for a definition    
            elif line.startswith('name:'):   
                name = line.strip().split('name: ')[1]    
                if current_go_id is not None:    
                    go_names[current_go_id] = name  
            elif line.startswith('def:'):    
                definition = line.strip().split('def:')[1]    
                definition = re.findall(r'"(.*?)"', definition)[0]    
                if current_go_id is not None:    
                    go_terms[current_go_id] = definition    
    return go_terms, go_names  

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=1, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True
    
def go_metrics(output_text, go_labels):
    
    # calculate the recall and precision of the predicted GO terms
    
    
    count_recall = 0
    for go in go_labels:
        if go in output_text:
            count_recall += 1
    recall = count_recall / len(go_labels)
    
    rouge_score = Rouge().get_scores(output_text, " ".join(go_labels), avg=True)['rouge-l']['r']
    
    return rouge_score


def main():
    ######### go information #########

    go_terms, go_names = parse_go_terms('/default/users/v-changma1/multi-modal-data/go-basic.obo')
    go_dag = GODag('/default/users/v-changma1/multi-modal-data/go-basic.obo')  

    # ========================================
    #            Loading Test Data
    # ========================================
    
    data = np.load('/default/users/v-changma1/multi-modal-data/GO/test_seq_label.npz', allow_pickle=True)
    test_data = data["data"]
    
    # ========================================
    #             Model Initialization
    # ========================================
    
    args = parse_args()
    cfg = Config(args)
    
    ckpt_last_name = cfg.model_cfg.ckpt.split('.')[0]
    file_name = '_'.join(ckpt_last_name.split('/')[-2:])
    file_name = "minigpt4/output_test/go_test_" + file_name+ ".txt"
    output_file = open(file_name, 'a')
        
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    vis_processor_cfg = cfg.datasets_cfg.swissprot.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    
    generator = ProteinTextGenerator(model, vis_processor, args.gpu_id)


    # ========================================
    #             Model Evaluation
    # ========================================
    
    prompts = [
        "Review the subsequent protein sequence and estimate its involvement in any {}: ",
        "Please examine the protein encoded by the amino acid sequence and describe its potential involvement in {}: ",
        "Could you analyze the protein corresponding to the amino acid sequence and offer insights on its {}?",
        "What {} is this protein involved in?",
        "Tell me about the {} of this protein.",
        "The {} of this protein include: ",
    ]

    location_prompts = [
        "Review the subsequent protein sequence and estimate its subcellular location: ",
        "Please examine the protein encoded by the amino acid sequence and describe its location within a cell: ",
        "Please identify the most probable subcellular location for the protein with the following amino acid sequence:",
        "What is the subcellular location of this protein?",
        "Based on the given protein sequence, predict the cellular compartment it would predominantly reside in: ",
        "The protein is located in: "
    ]


    all_scores_mf = []
    all_scores_cc = []
    all_scores_bp = []
    
    for i in tqdm(range(len(test_data))):
        item = test_data[i]
        sequence = item["sequence"]
        
        output_file.write(sequence + '\n')
        
        mf_labels = item["mf_labels"]
        cc_labels = item["cc_labels"]
        bp_labels = item["bp_labels"]
        
        mf_labels = [go_dag[x].id if x in go_dag else x for x in mf_labels]  
        cc_labels = [go_dag[x].id if x in go_dag else x for x in cc_labels]  
        bp_labels = [go_dag[x].id if x in go_dag else x for x in bp_labels]  
        
        mf_labels = [go_names[x] for x in mf_labels if x in go_names]
        cc_labels = [go_names[x] for x in cc_labels if x in go_names]
        bp_labels = [go_names[x] for x in bp_labels if x in go_names]
        

        best_score_mf = 0
        
        if len(mf_labels) > 0:
            output_file.write("MF GT: " + " ".join(mf_labels) + '\n')
            for p in prompts:
                p = p.format("molecular function")
                output_text = generator.generate([p], [sequence])
                score = go_metrics(output_text, mf_labels)
            
                if score > best_score_mf:
                    best_score_mf = score
                    
                output_file.write(str(score) + " " + output_text+ '\n')
                
        
        all_scores_mf.append(best_score_mf)
        
        best_score_bp = 0
        if len(bp_labels) > 0:
            output_file.write("BP GT: " + " ".join(bp_labels) + '\n')
            for p in prompts:
                p = p.format("biological process")
                output_text = generator.generate([p], [sequence])
                score = go_metrics(output_text, bp_labels)
            
                if score > best_score_bp:
                    best_score_bp = score
                output_file.write(str(score) + " " + output_text+ '\n')
        
        all_scores_bp.append(best_score_bp)
        
        best_score_cc = 0
        if len(cc_labels) > 0:
            output_file.write("CC GT: " + " ".join(cc_labels) + '\n')
            for p in location_prompts:
                output_text = generator.generate([p], [sequence])
                score = go_metrics(output_text, cc_labels)
                
                if score > best_score_cc:
                    best_score_cc = score
                output_file.write(str(score) + " " + output_text+ '\n')
        all_scores_cc.append(best_score_cc)
        
    print("Average ROUGE-L score for MF: ", np.mean(np.array(all_scores_mf)))
    print("Average ROUGE-L score for BP: ", np.mean(np.array(all_scores_bp)))
    print("Average ROUGE-L score for CC: ", np.mean(np.array(all_scores_cc)))
    
    
    
    
if __name__ == "__main__":
    main()