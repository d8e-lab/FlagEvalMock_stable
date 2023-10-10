MODEL=llama
LLAMA_BASE=/data/LLM/llama-7b-hf/
{
    CUDA_VISIBLE_DEVICES=6 python main_v2.py --dataset-names='BoolQ' \
        --model-name $MODEL \
        --model-path $LLAMA_BASE \
        --tokenizer-path $LLAMA_BASE \
        --batch-size 8 
        # --verbose 
        # --no-save
} # > ./logs/llama_IMDB.txt    
