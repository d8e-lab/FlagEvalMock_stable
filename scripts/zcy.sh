# MODEL=llama2_repadapter
MODEL=llama2
LLAMA_BASE=/mnt/SFT_store/Linksoul-llama2-7b
# LLAMA_BASE=/mnt/SFT_store/flageval_peft/outputs/lora/2023-10-14_17-11-51_success
current_datetime=$(date +"%m%d_%H_%M_%S")
export CUDA_VISIBLE_DEVICES=0,1,3,5,6,7
# EPRSTMT,TNEWS,OCNLI,BUSTM TruthfulQA
torchrun --nproc-per-node 6 --master-port 25000 main_v2dist.py --dataset-names="EPRSTMT,TNEWS,OCNLI,BUSTM" \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 1 \
    --verbose 
    # > "./logs/boolq_lora.log" 
