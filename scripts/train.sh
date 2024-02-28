#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
# export CUDA_VISIBLE_DEVICES=4,5,6,7
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=5333
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/data/megatron_ckpt/test_prune_3B
VOCAB_FILE=/data/hf_models/Llama-2-7b-hf/tokenizer.model
DATA_PATH=/data/v2_data/pile/bins/11-14-pile
LOAD_PATH=/data/megatron_ckpt/llama7B_pruned_3B_tp
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
    --num-layers  28 \
    --hidden-size 3072 \
    --num-attention-heads 24 \
    --ffn-hidden-size 8192 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 8 \
    --global-batch-size 1024 \
    --lr 2.5e-5 \
    --train-iters 12500 \
    --lr-decay-iters 12500 \
    --lr-warmup-iters 50 \
    --lr-decay-style cosine \
    --min-lr 1e-5 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --bf16 \
    --swiglu \
    --log-interval 10 \
    --exit-on-missing-checkpoint \
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
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model $VOCAB_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 250 \
    --eval-interval 100 \
    --eval-iters 1
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --tensorboard-dir $CHECKPOINT_PATH  | tee $CHECKPOINT_PATH/$NODE_RANK.log


