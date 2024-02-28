#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=4842
NNODES=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
NODE_RANK=0

CHECKPOINT_PATH=/data/megatron_ckpt/llama7B_pruned_3B_tp4
VOCAB_FILE=/data/Megatron-LM-main/baichuan_tokenizer/tokenizer.model
DATA_PATH=/data/v2_data/megatron_new/130B/130B_zhen
LOAD_PATH="/data/Megatron-LM-main/ckpts/Llama2-7b-tp4"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --sequence-parallel \
    --num-layers 32 \
    --hidden-size 4096 \
    --kv-channels 128 \
    --num-attention-heads 32 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 2 \
    --global-batch-size 64 \
    --lr 2.5e-5 \
    --train-iters 4800 \
    --lr-decay-iters 4672 \
    --lr-warmup-iters 128 \
    --lr-decay-style cosine \
    --min-lr 2.5e-5 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --fp16 \
    --log-interval 10 \
    --exit-on-missing-checkpoint \
    --use-checkpoint-args \
    --no-load-optim \
    --no-load-rng \
    --initial-loss-scale 131072 \
    --untie-embeddings-and-output-weights \
    --use-rotary-position-embeddings \
    --normalization RMSNorm \
    --no-position-embedding \
    --no-masked-softmax-fusion \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model $VOCAB_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 100 \
    --eval-interval 100 \
    --eval-iters 1
"
PRUNE_ARGS="
    --hidden_size_remain 3072 \
    --num_attention_heads_remain 24 \
    --ffn_hidden_size_remain 8192 \
    --drop_layers 7,13,18,24 \
    --prune_type balance 
"

torchrun $DISTRIBUTED_ARGS prune_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --is_prune \
    $PRUNE_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $LOAD_PATH \ 
    --tensorboard-dir $CHECKPOINT_PATH 


