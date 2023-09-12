MODEL=llama2
LLAMA_BASE=/data/LLM/Llama-2-7b-hf/
current_datetime=$(date +"%m%d_%H_%M_%S")
CUDA_VISIBLE_DEVICES=0 python main_v2.py --dataset-names="BoolQ" \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 2 \
    --verbose \
    > "./logs/merge_${current_datetime}_${MODEL}.log" 2>&1