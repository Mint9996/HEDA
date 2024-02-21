SRC=src
TGT=tgt

for i in 1 2 3
do
  DATA=../train_data/pro${i}  #data_file
  SPM=../sentencepiece/build/src/spm_encode
  MODEL=../llama_shared_data/tokenizer.model

  mkdir ${DATA}/data-bin
  cd ${DATA}

  ${SPM} --model=${MODEL} < train${i}.${SRC} > ${DATA}/data-bin/train.spm.${SRC}
  ${SPM} --model=${MODEL} < train${i}.${TGT} > ${DATA}/data-bin/train.spm.${TGT}

  python ../alpaca_lora/src/preprocess.py \
    --user-dir ../alpaca_lora/src \
    --task llama_task \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --trainpref ${DATA}/data-bin/train.spm \
    --destdir ${DATA}/data-bin/ \
    --srcdict ../alpaca_lora/scripts/assert/dict.txt \
    --tgtdict ../alpaca_lora/scripts/assert/dict.txt \
    --workers 20 \

done



