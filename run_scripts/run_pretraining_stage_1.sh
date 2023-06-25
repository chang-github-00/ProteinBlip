cd ..

NUM_GPU=4

CUDA_VISIBLE_DEVICES=0,1,2,5 torchrun --nproc-per-node $NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml