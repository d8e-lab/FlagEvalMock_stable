
MODEL=LinkSoul_sft
LLAMA_BASE=/data/LLM/LinkSoul_checkpoints_llama2_chat_2/
{
    CUDA_VISIBLE_DEVICES=2 python main_v2.py --dataset-names='IMDB'\
        --model-name $MODEL \
        --model-path $LLAMA_BASE \
        --tokenizer-path $LLAMA_BASE \
        --batch-size 8 \
        --verbose 
}> ./logs/linksoul_sft_IMDB.log