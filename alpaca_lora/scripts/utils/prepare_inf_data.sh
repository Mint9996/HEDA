# SRC=src
# TGT=tgt

# DATA=/opt/data/private/data/llama/llama_raw/inf/
# SPM=/opt/data/private/code/sentencepiece/build/src/spm_encode
# MODEL=/opt/data/private/data/llama/tokenizer.model

# ${SPM} --model=${MODEL} < ${DATA}/test.${SRC} > ${DATA}/test.spm.${SRC}

# # cp ${DATA}/test.spm.${SRC} ${DATA}/test.spm.${TGT}

# python alpaca_lora/src/preprocess.py \
#   --user-dir alpaca_lora/src \
#   --task llama_task \
#   --source-lang ${SRC} \
#   --target-lang ${TGT} \
#   --testpref ${DATA}/test.spm \
#   --destdir ${DATA}/data-bin \
#   --srcdict alpaca_lora/scripts/dict.txt \
#   --tgtdict alpaca_lora/scripts/dict.txt \
#   --workers 40 \


SRC=src
TGT=tgt

DATA=/opt/data/private/xys/llama_shared_data/llama_raw/test_data/
SPM=/opt/data/private/xys/efficient_alpaca-main/sentencepiece/build/src/spm_encode
MODEL=/opt/data/private/xys/llama_shared_data/tokenizer.model

${SPM} --model=${MODEL} < ${DATA}/test.${SRC} > ${DATA}/test.spm.${SRC}
${SPM} --model=${MODEL} < ${DATA}/test.${TGT} > ${DATA}/test.spm.${TGT}

# cp ${DATA}/test.spm.${SRC} ${DATA}/test.spm.${TGT}

python alpaca_lora/src/preprocess.py \
  --user-dir alpaca_lora/src \
  --task llama_task \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --testpref ${DATA}/test.spm \
  --destdir ${DATA}/data-bin \
  --srcdict alpaca_lora/scripts/assert/dict.txt \
  --tgtdict alpaca_lora/scripts/assert/dict.txt \
  --workers 8 \


