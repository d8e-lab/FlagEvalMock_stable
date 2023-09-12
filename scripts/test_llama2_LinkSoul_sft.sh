
MODEL=LinkSoul_base
LLAMA_BASE=/data/LLM/Linksoul-llama2-7b/
CUDA_VISIBLE_DEVICES=1 python main_v2.py --dataset-names='GAOKAO2023' \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 8 
    # --verbose \
    # --no-save

