MODEL=llama2
# tokenizer=/mnt/store/internlm-20b/
LLAMA_BASE=/mnt/42_store/LLMs/LingoWhale-8B/
mkdir -p "./evaluation_results/$(basename ${LLAMA_BASE})"
current_datetime=$(date +"%m%d_%H_%M_%S")
mkdir -p "./logs/${MODEL}/$(basename ${LLAMA_BASE})"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
torchrun --master-port 29556 --nproc-per-node 6 main_v2dist.py --dataset-names="ALL" \
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 1 \
    --verbose \
    --saver-path "./evaluation_results/$(basename ${LLAMA_BASE})" \
    > "./logs/${MODEL}/$(basename ${LLAMA_BASE})/${current_datetime}.log" 2>&1