import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

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
    



def main():
    
    # ========================================
    #             Model Initialization
    # ========================================
    
    args = parse_args()
    cfg = Config(args)
    
    ckpt_last_name = cfg.model_cfg.ckpt.split('.')[0]
    file_name = '_'.join(ckpt_last_name.split('/')[-2:])
    file_name = "minigpt4/output_test/test_" + file_name+ ".txt"
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
    
    # protein_sequence = [
    #     "MGQSFNAPYEAIGEELLSQLVDTFYERVASHPLLKPIFPSDLTETARKQKQFLTQYLGGPPLYTEEHGHPMLRARHLPFPITNERADAWLSCMKDAMDHVGLEGEIREFLFGRLELTARHMVNQTEAEDRSS",
    # ]
    # query = [
    #     "The function of this protein is"
    # ]
    
    
    
    while(1):
        print("Please input a protein sequence:")
        sequence = input()
        print("Please input a query:")
        query = input()
        
        protein_sequence = [sequence]
        query = [query]
        
    
        output_text, _ = generator.generate(query, protein_sequence)
        
        print(output_text)
        
        output_file.write(sequence + '\n')
        output_file.write(query[0] + '\n')
        output_file.write(output_text)
        output_file.write('\n')
        output_file.write('\n')
    

if __name__ == "__main__":
    main()