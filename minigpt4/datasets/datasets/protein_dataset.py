import json
from minigpt4.datasets.datasets.base_dataset import BaseDataset



class TextProteinDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, ann_paths):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)
        self.sequence = []
        
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

        return {
            "text_input": ann["caption"],
            "chain": ann["sequence"],
            "id": ann["id"]
        }

    
    def load_annotation_sequences(self, ann_path):
        annotations = []
        with open(ann_path, "r") as f:
            for line in f:
                line = line.strip()
                item = json.loads(line)
                annotations.append(item)
        return annotations


