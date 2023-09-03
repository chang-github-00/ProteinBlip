import argparse
import os
import random
from tqdm import tqdm
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

from minigpt4.conversation.simple_generator import ProteinTextGenerator, TextProteinGenerator

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

    # vis_processor_cfg = cfg.datasets_cfg.swissprot.vis_processor.train
    # vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    
    generator = TextProteinGenerator(model, args.gpu_id)

    # ========================================
    #             Model Evaluation
    # ========================================
    
    # protein_sequence = [
    #     "MGQSFNAPYEAIGEELLSQLVDTFYERVASHPLLKPIFPSDLTETARKQKQFLTQYLGGPPLYTEEHGHPMLRARHLPFPITNERADAWLSCMKDAMDHVGLEGEIREFLFGRLELTARHMVNQTEAEDRSS",
    # ]
    # query = [
    #     "The function of this protein is"
    # ]
    
    
    
    # while(1):
    #     print("Please input a query:")
    #     query = input()
        
    #     query = [query]
        
    #     if True:
    #         output_text = generator.generate(query)
            
    #         print(output_text)
            
    #         print("Please judge the quality of this answer, give ground truth if necessary:")
    #         gt = input()
            
    #         output_file.write(query[0] + '\n')
    #         output_file.write(output_text+ '\n')
    #         output_file.write("GT: " + gt + '\n')
    #         output_file.write('\n')
    #         output_file.write('\n')
    #     else:
    #         print("Error: Please input a valid protein sequence and query.")
    #         continue    

    all_queries = {
        "A0A1J4YT16_9PROT":"Generate a protein sequence that satisfies this requirement: rubisco with enhanced enzyme activity, leading to faster carboxylation rates and potentially improving the conversion of CO2 into biomass. ",
        "B1LPA6_ECOSM":"Generate a horismate mutase protein that enhance enzyme activity:",
        "AAV":"We aim to find diverse and novel AAV capsids capable of immune evasion and packaging their own genomes. Adeno-associated virus (AAV) capsids have shown clinical promise as delivery vectors for gene therapy. Generate a protein sequence that satisfies this requirement:",
        "Null": ""
    }
    
    for key in tqdm(all_queries):
        query = [all_queries[key]]
        output_file = open("minigpt4/output_dms/test_" + key + ".txt", 'a')
        for i in tqdm(range(20)):
            output_text = generator.generate(query)
            output_file.write(output_text + '\n')
            output_file.flush()
        
        output_file.close() 
        
    
        
        
    
if __name__ == "__main__":
    main()