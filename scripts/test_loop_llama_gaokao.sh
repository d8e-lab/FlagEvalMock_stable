MODEL=llama
LLAMA_BASE=/data/LLM/llama-7b-hf/
COUNT=1
while [ $COUNT -le 10 ]; do
    echo "Count: $COUNT"
    CUDA_VISIBLE_DEVICES=4 python main_v2.py --dataset-names='GAOKAO2023' \
        --model-name $MODEL \
        --model-path $LLAMA_BASE \
        --tokenizer-path $LLAMA_BASE \
        --batch-size 8 
        # --verbose \
        # --no-save
    COUNT=$((COUNT + 1))
done