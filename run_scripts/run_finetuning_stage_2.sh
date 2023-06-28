cd ..

NUM_GPU=4

CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc-per-node $NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml