MODEL=llama2
LLAMA_BASE=/mnt/SFT_store/xxw/sampled_dataset_500/checkpoint-300
current_datetime=$(date +"%m%d_%H_%M_%S")
CUDA_VISIBLE_DEVICES=0,1,2.3,4,5
torchrun --nproc-per-node 6 main_v2dist.py --dataset-names="ALL" \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 1 \
    --verbose \
    > "./logs/merge_${current_datetime}_${MODEL}.log" 2>&1