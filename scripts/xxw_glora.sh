# MODEL=llama2_repadapter
MODEL=llama2_glora
LLAMA_BASE=/mnt/SFT_store/Linksoul-llama2-7b
# CONFIG_NO=0
# LLAMA_BASE=/mnt/SFT_store/flageval_peft/outputs/lora/2023-10-14_17-11-51_success
current_datetime=$(date +"%m%d_%H_%M_%S")
export CUDA_VISIBLE_DEVICES=6,7
# EPRSTMT,TNEWS,OCNLI,BUSTM TruthfulQA
dataset="EPRSTMT"

# CONFIG_NO=9
for CONFIG_NO in {0..9}; do
    saver_path="./evaluation_results/glora/$dataset/$current_datetime$CONFIG_NO"
    mkdir -p $saver_path
    cp scripts/xxw_glora.sh $saver_path
    cp main_v2dist_glora2.py $saver_path
    torchrun --nproc-per-node 2 --master_port=25577 main_v2dist_glora2.py --dataset-names=$dataset \
        --model-name $MODEL \
        --model-path $LLAMA_BASE \
        --tokenizer-path $CONFIG_NO \
        --batch-size 1 \
        --verbose \
        --glora_param_path "/mnt/SFT_store/flageval_peft/outputs/glora/2023-10-16_18-03-57_success/final.pt" \
        --glora_config_path "/mnt/SFT_store/flageval_peft/outputs/glora/search/2023-11-28_08-59-30_eprstmt/checkpoint-16.pth.tar" \
        --saver_path $saver_path
    sleep 10  # optional: sleep for 10 seconds between runs
done