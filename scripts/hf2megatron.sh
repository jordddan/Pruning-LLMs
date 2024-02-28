TP=8
HF_FORMAT_DIR=/data/hf_models/Llama-2-13b-hf
MEGATRON_FORMAT_DIR=/data/Megatron-LM-main/ckpts/Llama2-13b
TOKENIZER_MODEL=/data/hf_models/tokenizer.model

python tools/checkpoint/util.py \
--model-type GPT \
--loader llama2_hf \
--saver megatron \
--target-tensor-parallel-size ${TP} \
--load-dir ${HF_FORMAT_DIR} \
--save-dir ${MEGATRON_FORMAT_DIR} \
--tokenizer-model ${TOKENIZER_MODEL}