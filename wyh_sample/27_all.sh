MODEL=llama2_lora
# LLAMA_BASE=/mnt/SFT_store/wyh/weights/llama2-checkpoints-plus-sft-v2_checkpoint4000_lora_checkpoint31200_merge_ad10072223
LLAMA_BASE=$1
current_datetime=$(date +"%m%d_%H_%M_%S")
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7
# torchrun --nproc-per-node 6 main_v2dist.py --dataset-names="EPRSTMT,TNEWS,OCNLI,BUSTM" \ 
torchrun --nproc-per-node 4 --master-port 23342 /mnt/SFT_store/3090_eval/FlagEvalMock_stable/main_v2dist.py\
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 1
    # --verbose \
    # > "./logs/boolq_lora.log" 