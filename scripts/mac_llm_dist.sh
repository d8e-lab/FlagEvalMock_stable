# /mnt/store/llama2-plus-checkpoints/checkpoint-500
# /mnt/store/llama2-checkpoints/checkpoint-1500/
MODEL=mac_llm
STEP=142800
# LLAMA_BASE="/mnt/store/llama2-checkpoints-plus-continue_lora/checkpoint-${STEP}"
# LLAMA_BASE="/mnt/store/llama2-checkpoints-plus-longer_lora_merge"
LLAMA_BASE=/mnt/store/llama2-checkpoints-plus-continue_lora_merge
mkdir -p "./logs/${MODEL}/longer/"
mkdir -p "./logs/${MODEL}/continue/"
current_datetime=$(date +"%m%d_%H_%M_%S")
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
torchrun --nproc-per-node 6 main_v2dist.py --dataset-names='MMLU' \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 2 \
    --verbose \
    > "./logs/${MODEL}/continue/${current_datetime}.log" 2>&1

echo ${LLAMA_BASE} > "./logs/${MODEL}/continue/${current_datetime}.log"