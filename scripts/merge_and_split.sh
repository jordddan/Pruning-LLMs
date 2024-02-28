export CUDA_VISIBLE_DEVICES=4,5,6,7
python tools/checkpoint/util.py \
        --model-type GPT \
        --load-dir /data/megatron_ckpt/llama13B_pruned_7B_tp4 \
        --save-dir /data/megatron_ckpt/llama13B_pruned_7B_tp1 \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --megatron-path /data/LEO/megatron_prune

python tools/checkpoint/util.py \
        --model-type GPT \
        --load-dir /data/megatron_ckpt/llama13B_pruned_7B_tp1 \
        --save-dir /data/megatron_ckpt/llama13B_pruned_7B_tp1_new \
        --target-tensor-parallel-size 4 \
        --target-pipeline-parallel-size 1 \
        --megatron-path /data/LEO/megatron_prune