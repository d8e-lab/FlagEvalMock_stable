MODEL=InternLM
LLAMA_BASE=/mnt/SFT_store/LLM/InternLM-hf/
# LLAMA_BASE=$1
current_datetime=$(date +"%m%d_%H_%M_%S")
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export CUDA_VISIBLE_DEVICES=$1
# export CUDA_VISIBLE_DEVICES=3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
# torchrun --nproc-per-node 6 main_v2dist.py --dataset-names="EPRSTMT,TNEWS,OCNLI,BUSTM" \
# torchrun --nproc-per-node 8 --rdzv-endpoint 0.0.0.0:25234 /mnt/SFT_store/3090_eval/FlagEvalMock_stable/main_v2dist.py --dataset-names="ALL"\  TruthfulQA
# torchrun --nproc-per-node 1 --rdzv-endpoint 0.0.0.0:25233 /mnt/SFT_store/3090_eval/FlagEvalMock_stable/main_v2dist.py --dataset-names="EPRSTMT"\ CLUEWSC
# torchrun --nproc-per-node 1 --rdzv-endpoint 0.0.0.0:25233 /mnt/SFT_store/3090_eval/FlagEvalMock_stable/main_v2dist.py --dataset-names="CSL"\

# torchrun --nproc-per-node 1 --rdzv-endpoint 0.0.0.0:25234 /mnt/SFT_store/3090_eval/FlagEvalMock_stable/main_v2dist.py --dataset-names="TNEWS"\
# torchrun --nproc-per-node 1 --rdzv-endpoint 0.0.0.0:25233 /mnt/SFT_store/3090_eval/FlagEvalMock_stable/main_v2dist.py --dataset-names="CSL"\ RAFT
torchrun --nproc-per-node $2 --rdzv-endpoint 0.0.0.0:$3 /mnt/SFT_store/3090_eval/FlagEvalMock_stable/main_v2dist.py --dataset-names="RAFT"\
    --model-name $MODEL \
    --model-path $LLAMA_BASE \
    --tokenizer-path $LLAMA_BASE \
    --batch-size 1 \
    --verbose \
    --no-save \

    # > "./logs/boolq_lora.log" 