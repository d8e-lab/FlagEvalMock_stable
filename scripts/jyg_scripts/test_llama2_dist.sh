MODEL=llama2
LLAMA_BASE=/mnt/store/llama2-checkpoints-plus-sft-v3/checkpoint-9750
CUDA_VISIBLE_DEVICES=4,5,7
torchrun --nproc-per-node 3 main_v2dist.py --dataset-names="Chinese_MMLU" \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 8 \
    #--verbose \
    