MODEL=llama2
tokenizer=/mnt/store/Llama-2-20b-hf/
LLAMA_BASE=/mnt/store/llama2-20b-checkpoints/checkpoint-2000
current_datetime=$(date +"%m%d_%H_%M_%S")
export CUDA_VISIBLE_DEVICES=6,7
mkdir -p "./evaluation_results/$(basename ${LLAMA_BASE})"
torchrun --nproc-per-node 2 main_v2dist.py --dataset-names="ALL" \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $tokenizer \
    --batch-size 1 \
    --verbose \
    --saver-path "./evaluation_results/$(basename ${LLAMA_BASE})"
    # > "./logs/merge_${current_datetime}_${MODEL}.log" 2>&1