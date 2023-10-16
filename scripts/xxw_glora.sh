# MODEL=llama2_repadapter
MODEL=llama2_glora
LLAMA_BASE=/mnt/SFT_store/Linksoul-llama2-7b
CONFIG_NO=0
# LLAMA_BASE=/mnt/SFT_store/flageval_peft/outputs/lora/2023-10-14_17-11-51_success
current_datetime=$(date +"%m%d_%H_%M_%S")
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# EPRSTMT,TNEWS,OCNLI,BUSTM TruthfulQA
for CONFIG_NO in {3..49}; do
    torchrun --nproc-per-node 6 main_v2dist.py --dataset-names="EPRSTMT,TNEWS,OCNLI,BUSTM" \
        --model-name $MODEL \
        --model-path $LLAMA_BASE \
        --tokenizer-path $CONFIG_NO \
        --batch-size 1 \
        --verbose 
    sleep 10  # optional: sleep for 10 seconds between runs
done