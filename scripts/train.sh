#!/bin/bash
src=src
tgt=tgt

export CUDA_VISIBLE_DEVICES=0,1


data_dir=../train_data/data-bin
save_dir=../train_data/pro1/ckpt
llama_dir=../llama_shared_data/7B/megatron_2/model.pt
update_freq=128
world_size=2



python  ../alpaca_lora/src/train.py $data_dir \
    --reset-optimizer --reset-dataloader --reset-meters \
    --restore-file $llama_dir \
    --user-dir ../alpaca_lora/src \
    --max-target-positions 1024 \
    --model-parallel-size $world_size \
    --distributed-world-size $world_size \
    --task llama_task \
    --arch llama_7b \
    --lora-tuning \
    --criterion llama_loss \
    -s $src -t $tgt \
    --update-freq $update_freq \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr 2e-4 \
    --weight-decay 0.01 \
    --total-num-update 7000 --warmup-updates 200 \
    --no-progress-bar \
    --max-epoch 3 \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 128 \
    --batch-size 1 \
    --save-dir $save_dir