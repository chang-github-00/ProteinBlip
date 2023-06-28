# ProteinBlip
MINIGPT4 IMPLEMENTATION FOR PROTEIN


## Build Environment
conda env create -f environment.yml
source activate minigpt4

## Pretraining Stage 1
```
bash run_scripts/run_pretraining_stage_1.sh
```

## Pretraining Stage 2
```
bash run_scripts/run_finetuning_stage_2.sh
```

## Evaluation with Simple Query
```
bash run_scripts/run_eval.sh
```

