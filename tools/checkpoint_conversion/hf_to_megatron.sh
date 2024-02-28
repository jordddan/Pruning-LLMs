python tools/checkpoint_conversion/llama_checkpoint_conversion.py \
--load_path "/data/hf_models/llama-7b-stage1/" \
--save_path "/data/Megatron-LLaMA/temp_checkpoint_megatron" \
--target_tensor_model_parallel_size 8 \
--target_pipeline_model_parallel_size 4 \
--target_data_parallel_size 16 \
--target_params_dtype "bf16" \
--make_vocab_size_divisible_by 1 \
--print-checkpoint-structure \
--megatron-path "/data/Megatron-LM-main"