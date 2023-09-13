MODEL=chatglm2-6b
LLAMA_BASE=/data/LLM/chatglm2-6b/
current_datetime=$(date +"%m%d_%H_%M_%S")
# CUDA_VISIBLE_DEVICES=0\
torchrun --nproc-per-node 2 main_v2dist.py --dataset-names='BoolQ' \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 2 \
    > ./logs/chatglm2_BoolQ_${current_datetime}.log 2>&1

