
MODEL=llama
LLAMA_BASE=/data/LLM/llama-7b-hf/
CUDA_VISIBLE_DEVICES=$1 python main_v2.py --dataset-names="ALL"\
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 4 \
    > ./logs/merge_test.log 2>&1