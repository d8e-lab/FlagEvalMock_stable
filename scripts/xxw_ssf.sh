MODEL=llama2_ssf
# MODEL=llama2_glora
LLAMA_BASE=/mnt/SFT_store/Linksoul-llama2-7b
LOAD_PATH=/mnt/40_store/flageval_peft/outputs/ssf/2023-12-11_08-11-14_success/final.pt
# LLAMA_BASE=/mnt/SFT_store/flageval_peft/outputs/lora/2023-10-14_17-11-51_success
current_datetime=$(date +"%m%d_%H_%M_%S")
export CUDA_VISIBLE_DEVICES=0,1,2,3
# EPRSTMT,TNEWS,OCNLI,BUSTM TruthfulQA
torchrun --nproc-per-node 4 --master_port=25579 main_v2dist.py --dataset-names="EPRSTMT,TNEWS,OCNLI,BUSTM" \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LOAD_PATH \
    --batch-size 1 \
    --verbose 
    # > "./logs/boolq_lora.log" 