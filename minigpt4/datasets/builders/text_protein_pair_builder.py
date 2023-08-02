import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from minigpt4.datasets.datasets.protein_dataset import TextProteinDataset


@registry.register_builder("pdb")
class PDBBuilder(BaseDatasetBuilder):
    train_dataset_cls = TextProteinDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/pdb/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            ann_paths=build_info.storage,
        )

        return datasets
    
@registry.register_builder("swissprot")
class SwissProtBuilder(BaseDatasetBuilder):
    train_dataset_cls = TextProteinDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/swissprot/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            ann_paths=build_info.storage,
        )

        return datasets
    
@registry.register_builder("moldesign")
class MolInstructDesignBuilder(BaseDatasetBuilder):
    train_dataset_cls = TextProteinDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/moldesign/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            ann_paths=build_info.storage,
            instruction_split=True
        )

        return datasets
    

@registry.register_builder("molinstruct")
class MolInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = TextProteinDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/molinstruct/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            ann_paths=build_info.storage,
            instruction_split=True
        )

        return datasets
    
@registry.register_builder("function_instruct")
class FunctionInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = TextProteinDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/function-instruction/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            ann_paths=build_info.storage,
            instruction_split=True
        )

        return datasets
    
@registry.register_builder("domain_instruct")
class DomainInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = TextProteinDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/domain-instruction/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            ann_paths=build_info.storage,
            instruction_split=True
        )

        return datasets

@registry.register_builder("ec_instruct")
class EnzymeCommisionInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = TextProteinDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/ec-instruction/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            ann_paths=[build_info.storage1, build_info.storage2],
            instruction_split=True
        )

        return datasets
    

@registry.register_builder("go_instruct")
class GeneOntologyInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = TextProteinDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/go-instruction/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            ann_paths=[build_info.storage1, build_info.storage2],
            instruction_split=True
        )

        return datasets
    
@registry.register_builder("motif_instruct")
class MotifInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = TextProteinDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/motif-instruction/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            ann_paths=build_info.storage,
            instruction_split=True
        )

        return datasets
    
@registry.register_builder("dms_instruct")
class FitnessInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = TextProteinDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/dms-instruction/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            ann_paths=build_info.storage,
            instruction_split=True
        )

        return datasets

@registry.register_builder("ppi_instruct")
class ProteinProteinInteractionInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = TextProteinDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/ppi-instruction/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            ann_paths=build_info.storage,
            instruction_split=True
        )

        return datasets
    
@registry.register_builder("ssp_instruct")
class SecondaryStructureInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = TextProteinDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/ssp-instruction/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            ann_paths=build_info.storage,
            instruction_split=True
        )

        return datasets
