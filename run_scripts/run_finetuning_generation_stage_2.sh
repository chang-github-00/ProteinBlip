cd ..

NUM_GPU=3

CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc-per-node $NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_generation.yaml