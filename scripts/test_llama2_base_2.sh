MODEL=llama2
LLAMA_BASE=/data/LLM/Llama-2-7b-hf/
# COUNT=1
# while [ $COUNT -le 10 ]; do
# {
#     # echo "Count: $COUNT"
#     CUDA_VISIBLE_DEVICES=2 python main_v2.py --dataset-names='ALL' \
#         --model-name $MODEL \
#         --model-path $LLAMA_BASE \
#         --tokenizer-path $LLAMA_BASE \
#         --batch-size 8 \
#         --verbose
#         # --no-save
#     # COUNT=$((COUNT+1))
# }> ./logs/llama2_GAOKAO2023.log
# # done
# DATASET_LIST=("BoolQ" "MMLU" "TruthfulQA" "IMDB" "RAFT" "Chinese_MMLU" "C-Eval" "GAOKAO2023" "CSL" "ChID" "CLUEWSC" "EPRSTMT" "TNEWS" "OCNLI" "BUSTM") 
DATASET_LIST=("BoolQ" "MMLU" "TruthfulQA" "RAFT" "C-Eval" "GAOKAO2023")
# DATASET_LIST=("IMDB"  "CSL" "ChID" "CLUEWSC" "EPRSTMT" "TNEWS" "OCNLI" "BUSTM") 

for dataset in "${DATASET_LIST[@]}"; do
    current_datetime=$(date +"%m%d_%H_%M_%S")
    {
        CUDA_VISIBLE_DEVICES=6 python main_v2.py --dataset-names="$dataset" \
        --model-name $MODEL \
        --model-path $LLAMA_BASE \
        --tokenizer-path $LLAMA_BASE \
        --batch-size 8 \
        --verbose 
        # --no-save
    } > "./logs/new_dataset_log/${dataset}_${current_datetime}_${MODEL}.log" 2>> "./logs/new_dataset_log/${dataset}_${current_datetime}_${MODEL}_error.log"
done