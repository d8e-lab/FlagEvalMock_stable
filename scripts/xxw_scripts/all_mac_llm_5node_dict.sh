MODEL=mac_llm
STEP=(1200 2400 3600 4800)
for step in "${STEP[@]}"
do
{
    LLAMA_BASE="/mnt/SFT_store/xxw/5node_8gpu_test/checkpoint-${step}"
    echo $LLAMA_BASE
    current_datetime=$(date +"%m%d_%H_%M_%S")
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
    torchrun --nproc-per-node 6 main_v2dist.py --dataset-names="GAOKAO2023,CSL,ChID,CLUEWSC,EPRSTMT,TNEWS,OCNLI,BUSTM" \
        --model-name $MODEL \
        --model-path $LLAMA_BASE \
        --tokenizer-path $LLAMA_BASE \
        --batch-size 1 \
        --verbose \
        > "./logs/${MODEL}_5node_${step}_${current_datetime}.log" 2>&1
} done
# GAOKAO2023,CSL,ChID,CLUEWSC,EPRSTMT,TNEWS,OCNLI,BUSTM