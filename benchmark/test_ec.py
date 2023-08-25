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

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
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
    
def ec_metrics(output_text, ec_terms):
    def score(input, ground_truth):
        rouge = Rouge()  
        return rouge.get_scores(input, ground_truth, avg=True)['rouge-l']['r']
  
    # Calculate ROUGE scores , return the best score
    best = 0 
    for ec in ec_terms:
        rouge_l = score(output_text, ec)
        if rouge_l > best:
            best = rouge_l
    return best


def main():
    ######### EC information #########

    ec_terms = {}
    ec_names = {}
    
    with open("/default/users/v-changma1/multi-modal-data/benchmark_data/ec_reaction.txt", "r") as f:
        for line in f:
            line = json.loads(line.strip())
            id = line["ec_number"]
            ec_terms[id] = line["reaction"]
            ec_names[id] = line["ec_name"]

    # ========================================
    #            Loading Test Data
    # ========================================
    
    data = np.load('/default/users/v-changma1/multi-modal-data/EC/test_seq_label.npz', allow_pickle=True)
    test_data = data["data"]
    
    # ========================================
    #             Model Initialization
    # ========================================
    
    args = parse_args()
    cfg = Config(args)
    
    ckpt_last_name = cfg.model_cfg.ckpt.split('.')[0]
    file_name = '_'.join(ckpt_last_name.split('/')[-2:])
    file_name = "minigpt4/output_test/ec_test_" + file_name+ ".txt"
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
    
    prompt = [
        
        "Using the protein sequence supplied, identify and describe the enzymatic catalytic activity, with emphasis on the chemical reaction it accelerates",
        "Describe the enzymatic catalytic activity, with emphasis on the chemical reaction it accelerates",
        "Please evaluate the following protein sequence and provide an explanation of the enzyme's catalytic activity, including the chemical reaction it facilitates: ",
        "What is the catalytic activity of the enzyme?",
        "What is the catalytic activity of the enzyme? Please describe the chemical reaction it facilitates",
        "Examine the provided protein sequence and determine the catalytic activity of the enzyme it represents, focusing on the chemical reaction it promotes: ",
    ]


    all_scores = []
    
    for i in tqdm(range(len(test_data))):
        item = test_data[i]
        sequence = item["sequence"]
        ec_labels = item["ec_labels"]
        ec_labels = [ec for ec in ec_labels if '-' not in ec]
    
        ec_terms = [ec_terms[ec] for ec in ec_labels if ec in ec_terms]
        if len(ec_terms) == 0:
            continue
        else:
            best_score_p = 0
            output_file.write(sequence + '\n')
            output_file.write("GT: " + " ".join(ec_terms) + '\n')
            for p in prompt:
                output_text = generator.generate([p], [sequence])
                
                score = ec_metrics(output_text, ec_terms)
                if score > best_score_p:
                    best_score_p = score
                output_file.write(p + '\n')
                output_file.write(output_text+ " " + str(score) + '\n')
                
            output_file.write('\n')
                
            all_scores.append(best_score_p)
        
    
    print("Average ROUGE-L score: ", np.mean(np.array(all_scores)))
    
    
    
if __name__ == "__main__":
    main()