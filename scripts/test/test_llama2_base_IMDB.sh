MODEL=llama2
LLAMA_BASE=/data/LLM/Llama-2-7b-hf/

dataset="MMLU"
current_datetime=$(date +"%m%d_%H_%M_%S")
{
    CUDA_VISIBLE_DEVICES=3 python main_v2.py --dataset-names="$dataset" \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 2 
    # --no-save
    # --verbose 
} #> "./logs/${dataset}_${current_datetime}_llaba2.log" 2>> "./logs/${dataset}_${current_datetime}_llama2_error.log"