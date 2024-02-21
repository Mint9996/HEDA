SRC=de
TGT=en

DATA=/opt/data/private/data/efficient_alpaca-main/data/iwslt14deen
SPM=/opt/data/private/data/sentencepiece/build/src/spm_encode
MODEL=/opt/data/private/data/llama_shared_data/tokenizer.model

${SPM} --model=${MODEL} < ${DATA}/test.${SRC} > ${DATA}/test.spm.${SRC}
${SPM} --model=${MODEL} < ${DATA}/test.${TGT} > ${DATA}/test.spm.${TGT}

python alpaca_lora/src/preprocess.py \
  --user-dir alpaca_lora/src \
  --task llama_task \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --testpref ${DATA}/test.spm \
  --destdir ${DATA}/data-bin \
  --srcdict alpaca_lora/scripts/dict.txt \
  --tgtdict alpaca_lora/scripts/dict.txt \
  --workers 40 \