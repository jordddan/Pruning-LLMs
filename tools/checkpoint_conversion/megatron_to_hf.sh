python tools/checkpoint_conversion/llama_checkpoint_conversion.py \
--convert_checkpoint_from_megatron_to_transformers \
--load_path "/data/megatron_ckpt/test_pruned_3B_tp1" \
--save_path "/data/megatron_ckpt/pruned_3B-hf/3000" \
--target_params_dtype "float32" \
--make_vocab_size_divisible_by 1 \
--print-checkpoint-structure \
--megatron-path "/data/Megatron-LM-main"
