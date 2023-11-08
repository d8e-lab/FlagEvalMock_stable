MODEL=llama2
# tokenizer=/mnt/store/internlm-20b/
LLAMA_BASE="/data/zs/LLM_Weight/codellama-7b-hf/"
current_datetime=$(date +"%m%d_%H_%M_%S")
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
mkdir -p "./evaluation_results/$(basename ${LLAMA_BASE})"
torchrun --master-port 29555 --nproc-per-node 6 main_v2dist.py --dataset-names="ALL" \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 1 \
    --verbose \
    --saver-path "./evaluation_results/$(basename ${LLAMA_BASE})"
    # > "./logs/merge_${current_datetime}_${MODEL}.log" 2>&1