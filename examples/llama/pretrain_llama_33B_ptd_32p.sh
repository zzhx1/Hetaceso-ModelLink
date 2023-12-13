# This is an example: train llama using PTD.

# The number of parameters is not aligned
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=4
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TOKENIZER_PATH=./dataset/llama_tokenizer
DATA_PATH=./dataset/llama_text_document
CHECKPOINT_PATH=./ckpt
DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# Main script
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_llama.py \
       --DDP-impl local \
       --tensor-model-parallel-size 4 \
       --pipeline-model-parallel-size 4 \
       --num-layers 60 \
       --hidden-size 6656 \
       --ffn-hidden-size 17920 \
       --num-attention-heads 52 \
       --micro-batch-size 4 \
       --global-batch-size 512 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 50000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --tokenizer-name-or-path $TOKENIZER_PATH \
       --tokenizer-not-use-fast \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --initial-loss-scale 524288.0 \
       --sequence-parallel \
       --mlp-layer-fusion \
       --use-distributed-optimizer \
       --position-embedding-type rope \
       --normalization RMSNorm \
       --use-fused-rmsnorm \
       --use-flash-attn \
       --release-fp32-grad \
       --checkpoint-activations \
       --recompute-method custom \
       --recomputation-layer-num 3 0 0 0 \
       --fp16 | tee logs/train.log