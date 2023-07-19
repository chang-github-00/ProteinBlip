cd ..

NUM_GPU=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node $NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune_v2.yaml