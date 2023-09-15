MODEL=mac_llm
STEP=1200
LLAMA_BASE="/mnt/SFT_store/xxw/5node_8gpu_test/checkpoint-${STEP}"
current_datetime=$(date +"%m%d_%H_%M_%S")
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
torchrun --nproc-per-node 6 main_v2dist.py --dataset-names="ALL" \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 1 \
    --verbose \
    > "./logs/${MODEL}_5node_${STEP}_${current_datetime}.log" 2>&1