MODEL=mistral_moe
BASE="/mnt/82_store/xxw/models/Mixtral-8x7B-Instruct-v0.1"
current_datetime=$(date +"%m%d_%H_ %M_%S")
export CUDA_VISIBLE_DEVICES=0,1

python main_v2.py \
    --dataset-names="ALL" \
    --model-name $MODEL \
    --model-path $BASE \
    --tokenizer-path $BASE \
    --batch-size 1 \
    --verbose 