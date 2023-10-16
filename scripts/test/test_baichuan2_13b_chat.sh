MODEL=llama2
LLAMA_BASE=/mnt/store/Baichuan2-13B-Chat/
current_datetime=$(date +"%m%d_%H_%M_%S")
export CUDA_VISIBLE_DEVICES=6
# torchrun --nproc-per-node 1 main_v2dist.py --dataset-names='BoolQ' \
mkdir -p "./evaluation_results/$(basename ${LLAMA_BASE})"
torchrun --master-port 29501 --nproc-per-node 1 main_v2dist.py --dataset-names='MMLU' \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 1 \
    --saver-path "./evaluation_results/$(basename ${LLAMA_BASE})"
    # > ./logs/chatglm2_BoolQ_${current_datetime}.log 2>&1