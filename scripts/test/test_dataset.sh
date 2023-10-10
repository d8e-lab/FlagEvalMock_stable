
MODEL=LinkSoul_base
LLAMA_BASE=/data/LLM/Linksoul-llama2-7b/
CUDA_VISIBLE_DEVICES=0 python main_v2.py --dataset-names='RAFT' \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 8 \
    --verbose \
    --no-save

