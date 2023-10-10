MODEL=llama2
#LLAMA_BASE=/mnt/store/llama2-checkpoints-plus-sft-v2_checkpoint4000_lora_checkpoint31200_merge
LLAMA_BASE=/data/zyx/2030/mac_0.4_dsnt
STEP=140000
LOG_PATH="./logs/sparse/${MODEL}/$(basename ${LLAMA_BASE})"

mkdir -p ${LOG_PATH}
current_datetime=$(date +"%m%d_%H_%M_%S")
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc-per-node 4 main_v2dist.py --dataset-names="BoolQ" \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 2 \
    --verbose \
    >> "${LOG_PATH}/${current_datetime}.log" 2>&1