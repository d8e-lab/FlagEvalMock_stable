MODEL=chatglm2-6b
LLAMA_BASE=/data/LLM/chatglm2-6b/
# {
    CUDA_VISIBLE_DEVICES=0 python main_v2.py --dataset-names='BoolQ' \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 8 \
    --verbose \
    > ./logs/chatglm2_BoolQ.log 2>&1
    # --no-save
# }> ./logs/chatglm2_IMDB.log