# /mnt/store/llama2-plus-checkpoints/checkpoint-500
# /mnt/store/llama2-checkpoints/checkpoint-1500/
MODEL=mac_llm
LLAMA_BASE=/mnt/store/llama2-checkpoints-plus-freeze/checkpoint-45000
CUDA_VISIBLE_DEVICES=2 python main_v2.py --dataset-names='ALL' \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 2 \
    --verbose \
    >./logs/mac.log 2> ./logs/mac_error.log