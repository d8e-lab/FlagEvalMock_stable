MODEL=LinkSoul_base
LLAMA_BASE=/data/LLM/Linksoul-llama2-7b/
current_datetime=$(date +"%m%d_%H_%M_%S")
mkdir -p "./logs/${MODEL}"
CUDA_VISIBLE_DEVICES=0,1,2.3,4,5
torchrun --nproc-per-node 6 main_v2dist.py --dataset-names="ALL" \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 1 \
    --verbose \
    >> "./logs/${MODEL}/${current_datetime}.log" 2>&1