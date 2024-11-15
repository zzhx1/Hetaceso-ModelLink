#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=2500


GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer model path"
CKPT_LOAD_DIR="your model load ckpt path"

TP=8
PP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --sequence-parallel \
    --num-layers 40 \
    --hidden-size 5120 \
    --ffn-hidden-size 13696 \
    --num-attention-heads 40 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --seq-length 4096 \
    --disable-bias-linear \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 32 \
    --untie-embeddings-and-output-weights \
    --make-vocab-size-divisible-by 64 \
    --lr 1e-5 \
    --no-gradient-accumulation-fusion \
    --load ${CKPT_LOAD_DIR} \
    --train-iters 1000 \
    --lr-decay-style cosine \
    --attention-dropout 0.0 \
    --position-embedding-type alibi \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --attention-softmax-in-fp32 \
    --min-lr 1e-7 \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 1024.0 \
    --adam-beta2 0.95 \
    --adam-eps 1.0e-5 \
    --no-load-optim \
    --no-load-rng \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 1 
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --jit-compile \
    --save $CKPT_SAVE_DIR \
    | tee logs/train_baichuan_13b.log
