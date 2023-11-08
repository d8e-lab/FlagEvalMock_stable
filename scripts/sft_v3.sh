MODEL=llama2
LLAMA_BASE=/mnt/store/llama2-checkpoints-plus-sft-v3/checkpoint-14000/
mkdir -p "./evaluation_results/$(basename ${LLAMA_BASE})"
current_datetime=$(date +"%m%d_%H_%M_%S")
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p "./logs/${MODEL}/$(basename ${LLAMA_BASE})"
torchrun --nproc-per-node 6 main_v2dist.py --dataset-names="ALL" \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 1 \
    --verbose \
    --saver-path "./evaluation_results/$(basename ${LLAMA_BASE})"
    >> "./logs/${MODEL}/$(basename ${LLAMA_BASE})/${current_datetime}.log" 2>&1
    