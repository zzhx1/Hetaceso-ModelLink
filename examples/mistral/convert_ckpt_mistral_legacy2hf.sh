# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python tools/checkpoint/convert_ckpt.py \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --model-type GPT \
    --loader megatron \
    --saver megatron \
    --save-model-type save_huggingface_llama \
    --load-dir ./model_weights/Mistral-legacy/ \
    --save-dir ./model_from_hf/Mistral-hf/    # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/Mistral-hf/mg2hg/