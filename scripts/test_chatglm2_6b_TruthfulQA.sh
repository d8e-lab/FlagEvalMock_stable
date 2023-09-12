MODEL=chatglm2-6b
LLAMA_BASE=/data/LLM/chatglm2-6b/
# {
    CUDA_VISIBLE_DEVICES=6 python main_v2.py --dataset-names='TruthfulQA' \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 8 
    --verbose \
    # --no-save
# }> ./logs/chatglm2_IMDB.log