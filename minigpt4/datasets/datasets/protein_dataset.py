import json
from minigpt4.datasets.datasets.base_dataset import BaseDataset



class TextProteinDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, ann_paths, instruction_split=False):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)
        self.sequence = []
        self.instruction_split = instruction_split
        if isinstance(ann_paths, str):
            ann_paths = [ann_paths]
            
        for ann_path in ann_paths:
            annotations = self.load_annotation_sequences(ann_path)
            self.annotation.extend(annotations)

        self.vis_processor = vis_processor    # vis_processor is any processor that can process modality other than text
        self.text_processor = text_processor

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):

        ann = self.annotation[index]

        output = dict()
        
        if "caption" in ann:
            output["text_input"] = ann["caption"]
        # if "prompt" in ann:
        #     output["prompt"] = self.text_processor(ann["prompt"])
        for attr in ann:
            if "sequence" in attr:
                new_attr = attr.replace("sequence", "chain")
                output[new_attr] = self.vis_processor(ann[attr])
        
        for attr in ann:
            if attr in ["id", "prompt", "instruction_split", "label", "mode"]:
                output[attr] = str(ann[attr]) # note that sometimes json will load "0","1" as 0/1, resulting in collation error
        
        if self.instruction_split:
            output["instruction_split"] = "True"
        
        output["index"] = index
        return output
    
    def load_annotation_sequences(self, ann_path):
        annotations = []
        with open(ann_path, "r") as f:
            for line in f:
                line = line.strip()
                try:
                    item = json.loads(line)
                    annotations.append(item)
                except:
                    pass
        return annotations


