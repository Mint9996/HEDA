#!/bin/bash
src=src
tgt=tgt
export CUDA_VISIBLE_DEVICES=0,1

data_dir=../test_data/pro1/data-bin
llama_dir=../llama_shared_data/7B/megatron_2
save_model=../train_data/pro1/ckpt/checkpoint5-model_part-0.pt
bpe_dir=../llama_shared_data/tokenizer.model
res_dir=../res/${dataset}
mkdir $res_dir
gen_dir=$res_dir/${dataset}_pro1_epo5.res
world_size=2



torchrun --master_port 28770 --nproc_per_node $world_size ../alpaca_lora/src/generate.py $data_dir \
    --user-dir ../alpaca_lora/src/ \
    --task llama_task \
    --model-parallel-size $world_size \
    --distributed-world-size $world_size \
    --lora-model-inf $save_model \
    --arch llama_7b \
    --lora-tuning \
    -s $src -t $tgt \
    --gen-subset test \
    --bpe 'sentencepiece' --sentencepiece-model $bpe_dir \
    --path $llama_dir/model.pt \
    --seed 1 \
    --required-batch-size-multiple 1 \
    --batch-size 10 \
    --skip-invalid-size-inputs-valid-test \
    --beam 1  --temperature 0.8 > $gen_dir


