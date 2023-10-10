
# /mnt/store/llama2-checkpoints-plus-sft-v2_checkpoint4000_lora_checkpoint31200_merge/
MODEL=llama2
# LLAMA_BASE=/mnt/SFT_store/xxw/outputs/all_peft/lora_continue/2023-09-18_04-02-42_merge
# LLAMA_BASE=/mnt/SFT_store/xxw/outputs/all_peft/lora_longer/2023-09-18_03-46-44_merge
# LLAMA_BASE=/mnt/SFT_store/xxw/outputs/all_peft/lora_longer/2023-09-18_03-46-44_merge
# STEP=140000
# MODEL_PATH="llama2-plus-checkpoints-freeze/checkpoint-${STEP}"
# MODEL_PATH="/mnt/store/llama2-checkpoints-plus-sft-v2/checkpoint-4000"
# LLAMA_BASE="/mnt/store/selected-checkpoint/${MODEL_PATH}"
# LLAMA_BASE="/mnt/store/llama2-checkpoints-plus-sft-v2/checkpoint-4000"  
# LLAMA_BASE=/mnt/store/llama2-checkpoints-plus-sft-v2_checkpoint4000_lora_checkpoint31200_merge
# LLAMA_BASE=/mnt/store/selected-checkpoint/llama2-plus-checkpoints-freeze/checkpoint-20000
LLAMA_BASE=/mnt/store/llama2-checkpoints-plus-sft-v2_checkpoint4000_lora_checkpoint31200_merge/
LOG_PATH="./logs/${MODEL}/$(basename ${LLAMA_BASE})"
mkdir -p ${LOG_PATH}
current_datetime=$(date +"%m%d_%H_%M_%S")
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc-per-node 6 main_v2dist.py --dataset-names="ALL" \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 2 \
    --verbose \
    >> "${LOG_PATH}/${current_datetime}.log" 2>&1