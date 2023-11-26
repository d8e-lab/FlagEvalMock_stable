MODEL=llama2_repadapter
# MODEL=llama2_glora
# LLAMA_BASE=/mnt/store/llama2-checkpoints-plus-sft-v3/checkpoint-9750
LLAMA_BASE=/mnt/store/llama2-checkpoints-plus-sft-v3/checkpoint-9750
REPADAPTER_PATH=/home/xmu/test/flageval_peft/outputs/repadapter/llama2-checkpoints-plus-sft-v3_checkpoint-9750_repadapter_plus_data1018/final.pt
# LLAMA_BASE=/mnt/SFT_store/flageval_peft/outputs/lora/2023-10-14_17-11-51_success
current_datetime=$(date +"%m%d_%H_%M_%S")
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
# EPRSTMT,TNEWS,OCNLI,BUSTM TruthfulQA
torchrun --nproc-per-node 6 --master_port=25579 main_v2dist.py --dataset-names="ALL" \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $REPADAPTER_PATH \
    --batch-size 1 \
    --verbose 
    # > "./logs/boolq_lora.log" 

    # 