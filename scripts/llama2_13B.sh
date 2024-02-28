#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=5333
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/data/LEO/megatron_prune/ckpts/Llama2-13b-tp8
VOCAB_FILE=/data/hf_models/Llama-2-7b-hf/tokenizer.model
DATA_PATH=/data/v2_data/pile/bins/11-14-pile

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --sequence-parallel \
    --num-layers 40 \
    --hidden-size 5120 \
    --num-attention-heads 40 \
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
    --bf16 \
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
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model $VOCAB_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 100 \
    --eval-interval 3000 \
    --eval-iters 20
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save /data/megatron_ckpt/llama_test \
    --tensorboard-dir $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH


