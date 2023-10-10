MODEL=llama2_lora
LLAMA_BASE=/mnt/SFT_store/xxw/outputs/all_peft/item5-checkpoint-9750
current_datetime=$(date +"%m%d_%H_%M_%S")
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# eprstmt tnews ocnli bustm TruthfulQA
torchrun --nproc-per-node 8 main_v2dist.py --dataset-names="ALL" \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 1 \
    --verbose \
    # > "./logs/boolq_lora.log" 