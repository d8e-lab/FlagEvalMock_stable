# MODEL=llama2_repadapter
MODEL=llama2
LLAMA_BASE=/mnt/40_store/xxw/models/answer_model/
# LLAMA_BASE=/mnt/SFT_store/xxw/outputs/llama2-checkpoints-plus-longer/checkpoint-27000/merged/
# LLAMA_BASE=/home/xmu/test/flageval_peft/outputs/repadapter/llama2-checkpoints-plus-sft-v3_checkpoint-9750_repadapter_plus_data1018/
# LLAMA_BASE=/data/wyh/llama2-checkpoints-plus-sft-v3_checkpoint-9750_repadapter_data1018/merged/
# LLAMA_BASE=/mnt/SFT_store/flageval_peft/outputs/lora/2023-10-14_17-11-51_success
current_datetime=$(date +"%m%d_%H_%M_%S")
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# EPRSTMT,TNEWS,OCNLI,BUSTM TruthfulQA ALL
torchrun --nproc-per-node 6 --master_port=25579 main_v2dist.py --dataset-names="ALL" \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 1 \
    --verbose 
    # > "./logs/boolq_lora.log" 