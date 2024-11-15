#!/bin/bash

export COMBINED_ENABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# fixed Model related configuration here, pls not overlap with json config
HIDDEN_SIZE=1024
NUM_ATTENTION_HEADS=16
SEQ_LENGTH=2048
MAX_POSITION_EMBEDDINGS=$SEQ_LENGTH
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=16


DATA_PATH="/workspace/experiment123/ModelLink/dataset/gpt_text_sentence"
VOCAB_FILE="/workspace/experiment123/ModelLink/vocab_file/gpt2-vocab.json"
MERGE_FILE="/workspace/experiment123/ModelLink/vocab_file/gpt2-merges.txt"


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

DATA_ARGS="
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --mock-data \
"

# Model related configuration here, pls not overlap with json config
GPT_ARGS="
    --no-async-tensor-model-parallel-allreduce \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --lr 0.00015 \
    --train-iters 20 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --tokenizer-type GPT2BPETokenizer \
    --use-flash-attn \
    --use-mcore-models \
    --transformer-impl local \
"

FLEX_ARGS="
    --flexpipe-config /workspace/experiment123/ModelLink/examples/aceso/pretrain_test/test_pretrain.json \
    --log-path  /workspace/experiment123/ModelLink/examples/aceso/pretrain_test/logs \
"
OUTPUT_ARGS="
    --log-interval 1
    --eval-interval 100
    --eval-iters 3
"
mkdir -p logs
mkdir -p logs/csv

torchrun $DISTRIBUTED_ARGS \
    pretrain_gpt.py \
    $GPT_ARGS \
    $FLEX_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --jit-compile \
    --distributed-backend nccl 2>&1 | tee ./pretrain_gpt3_175B_8layers.log
