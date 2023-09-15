MODEL=llama2
LLAMA_BASE=/data/LLM/Llama-2-7b-hf/
current_datetime=$(date +"%m%d_%H_%M_%S")
CUDA_VISIBLE_DEVICES=0,1,2.3,4,5,6,7
torchrun --nproc-per-node 8 main_v2dist.py --dataset-names="ALL" \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 1 \
    --verbose \
    > "./logs/merge_${current_datetime}_${MODEL}.log" 2>&1