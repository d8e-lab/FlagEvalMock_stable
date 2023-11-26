
MODEL_PATH="/mnt/store/llama2-checkpoints-plus-sft-v3/checkpoint-9750/"
# MODEL_PATH="/mnt/40_store/wyh_SFT/outputs/checkpoint9750_data1114_trick1101/"
output="/mnt/SFT_store/3090_eval/FlagEvalMock_stable/chinese_subjective/siyuan_chat.csv"
# output="/mnt/SFT_store/3090_eval/FlagEvalMock_stable/chinese_subjective/siyuan_chat_sft.csv"
# export CUDA_VISIBLE_DEVICES=6
export CUDA_VISIBLE_DEVICES=7
python chinese_subjective/chinese_subjective.py \
        --model_name_or_path ${MODEL_PATH} \
        --output ${output} \
        --model_max_length 2048 \
        --sample_num 1 \
# python /home/xmu/test/flageval_peft/wyh_auxiliarydata/merge_model.py ${MODEL_PATH} ${output}