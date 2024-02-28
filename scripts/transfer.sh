
python tools/checkpoint/util.py \
        --model-type GPT \
        --load-dir /data/megatron_ckpt/test_prune_3B \
        --save-dir /data/megatron_ckpt/test_pruned_3B_tp1 \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 