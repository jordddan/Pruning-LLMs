# for i in $(seq -w 1 50); do
#     python tools/preprocess_data.py \
#         --input /data/v2_data/langchao/$(printf "%03d" $i).jsonl \
#         --output-prefix /data/v2_data/new/lc$(printf "%02d" $i) \
#         --tokenizer-type GPTSentencePieceTokenizer \
#         --tokenizer-model /data/Megatron-LM-main/baichuan_tokenizer/tokenizer.model \
#         --append-eod \
#         --workers 16
# done

# for i in $(seq -w 10 29); do
#     python tools/preprocess_data.py \
#         --input /data/v2_data/pile/$(printf "%02d" $i).jsonl \
#         --output-prefix /data/v2_data/new/pile$(printf "%02d" $i) \
#         --tokenizer-type GPTSentencePieceTokenizer \
#         --tokenizer-model /data/Megatron-LM-main/baichuan_tokenizer/tokenizer.model \
#         --append-eod \
#         --workers 16
# done


# python tools/preprocess_data.py \
#     --input /data/v2_data/distill_data/deepseek/lc_01_p128_t2048_wieos.jsonl \
#     --output-prefix /data/v2_data/distill_data/bin_data/v1/lc \
#     --tokenizer-type Llama2Tokenizer \
#     --tokenizer-model /data/hf_models/chinese-llama2-13b/tokenizer.model \
#     --append-eod \
#     --workers 16

# python tools/preprocess_data.py \
#     --input /data/v2_data/distill_data/deepseek/pile_01_p128_t2048_wieos.jsonl \
#     --output-prefix /data/v2_data/distill_data/bin_data/v1/pile \
#     --tokenizer-type Llama2Tokenizer \
#     --tokenizer-model /data/hf_models/chinese-llama2-13b/tokenizer.model \
#     --append-eod \
#     --workers 16

python tools/preprocess_data.py \
    --input /data/v2_data/pile/10_small.jsonl \
    --output-prefix /data/v2_data/llama2_bin/pile_10_small  \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model /data/hf_models/Llama-2-7b-hf/tokenizer.model \
    --append-eod \
    --workers 16