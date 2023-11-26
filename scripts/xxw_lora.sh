# MODEL=llama2_repadapter
MODEL=llama2_lora
LLAMA_BASE=/mnt/SFT_store/flageval_peft/outputs/lora/2023-10-24_06-21-19_success
# LLAMA_BASE=/mnt/SFT_store/flageval_peft/outputs/lora/2023-10-14_17-11-51_success
current_datetime=$(date +"%m%d_%H_%M_%S")
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# EPRSTMT,TNEWS,OCNLI,BUSTM TruthfulQA
torchrun --nproc-per-node 6 --master_port=25579 main_v2dist.py --dataset-names="EPRSTMT,TNEWS,OCNLI,BUSTM" \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 1 \
    --verbose 
    # > "./logs/boolq_lora.log" 