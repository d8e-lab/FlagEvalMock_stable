MODEL=lora_mixtral
# BASE=/mnt/82_store/xxw/mixtral/outputs/lora/2024-01-23_17-49-14_success/checkpoint-164000/
BASE=/mnt/82_store/xxw/mixtral/outputs/lora/2024-01-27_13-22-07/checkpoint-6000
current_datetime=$(date +"%m%d_%H_ %M_%S")
export CUDA_VISIBLE_DEVICES=0,1

python main_v2.py \
    --dataset-names="ALL" \
    --model-name $MODEL \
    --model-path $BASE \
    --tokenizer-path $BASE \
    --batch-size 1 \
    --verbose 