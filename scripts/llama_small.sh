#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=wxhd03
MASTER_PORT=6000
NNODES=4
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/data/Megatron-LM-main/ckpts/small
VOCAB_FILE=/data/Megatron-LM-main/baichuan_tokenizer/tokenizer.model
DATA_PATH=/data/v2_data/megatron_new/130B/130B_zhen

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank 0 \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --sequence-parallel \
    --num-layers 8 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 16 \
    --global-batch-size 512 \
    --lr 2.5e-5 \
    --train-samples 4800000 \
    --lr-decay-samples 4672000 \
    --lr-warmup-samples 12800 \
    --lr-decay-style cosine \
    --min-lr 2.5e-5 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --bf16 \
    --log-interval 10 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-position-embedding \
    --use-rotary-position-embeddings \
    --normalization RMSNorm \
    --no-masked-softmax-fusion \
    --swiglu \
    --initial-loss-scale 131072 \
    --log-throughput \
    --use-flash-attn \
    --use-distributed-optimizer \
    --attention-softmax-in-fp32 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model $VOCAB_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10 \
    --eval-interval 3000 \
    --eval-iters 1
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --tensorboard-dir $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH


