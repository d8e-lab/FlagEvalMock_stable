
MODEL=llama
LLAMA_BASE=/data/LLM/llama-7b-hf/
CUDA_VISIBLE_DEVICES=$1 python main_v2.py --dataset-names="BoolQ"\
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 2 \
    > ./logs/merge_test.log 2>&1