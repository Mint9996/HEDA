SRC=src
TGT=tgt

DATA=/opt/data/private/xys/efficient_alpaca-main/mnli/llama_mnli
SPM=/opt/data/private/xys/efficient_alpaca-main/sentencepiece/build/src/spm_encode
MODEL=/opt/data/private/xys/llama_shared_data/tokenizer.model


${SPM} --model=${MODEL} < ${DATA}/train3.${SRC} > ${DATA}/promt3/train3.spm.${SRC}
${SPM} --model=${MODEL} < ${DATA}/train3.${TGT} > ${DATA}/promt3/train3.spm.${TGT}
${SPM} --model=${MODEL} < ${DATA}/valid3.${SRC} > ${DATA}/promt3/valid3.spm.${SRC}
${SPM} --model=${MODEL} < ${DATA}/valid3.${TGT} > ${DATA}/promt3/valid3.spm.${TGT}
${SPM} --model=${MODEL} < ${DATA}/test3.${SRC} > ${DATA}/promt3/test3.spm.${SRC}
${SPM} --model=${MODEL} < ${DATA}/test3.${TGT} > ${DATA}/promt3/test3.spm.${TGT}


python alpaca_lora/src/preprocess.py \
  --user-dir alpaca_lora/src \
  --task llama_task \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --trainpref ${DATA}/promt3/train3.spm \
  --validpref ${DATA}/promt3/valid3.spm \
  --testpref ${DATA}/promt3/test3.spm \
  --destdir ${DATA}/promt3/data-bin \
  --srcdict alpaca_lora/scripts/assert/dict.txt \
  --tgtdict alpaca_lora/scripts/assert/dict.txt \
  --workers 30 \
