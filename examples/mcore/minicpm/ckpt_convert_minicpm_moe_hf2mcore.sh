# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换，设置需要的并行配置
python tools/checkpoint/convert_ckpt.py \
    --model-type GPT \
    --loader hf_mcore \
    --saver mg_mcore \
    --use-mcore-models \
    --load-dir ./model_from_hf/MiniCPM-MoE-8x2B \
    --save-dir ./model_from_hf/MiniCPM-MoE-8x2B-mcore \
    --tokenizer-model ./model_from_hf/MiniCPM-MoE-8x2B/tokenizer.model \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 4 \
    --target-expert-parallel-size 2 \
    --model-type-hf minicpm-moe \
    --params-dtype bf16