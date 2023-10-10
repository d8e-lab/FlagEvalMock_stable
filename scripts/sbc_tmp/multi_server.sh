# ssh MAC_SUP_75
MODEL_LIST=(
/mnt/store/llama2-checkpoints-plus-sft-v3/checkpoint-12000/
/mnt/store/llama2-checkpoints-plus-sft-v3/checkpoint-12500/
/mnt/store/llama2-checkpoints-plus-sft-v3/checkpoint-13000/
/mnt/store/llama2-checkpoints-plus-sft-v3/checkpoint-13500/
/mnt/store/llama2-checkpoints-plus-sft-v3/checkpoint-14000/
/mnt/store/llama2-checkpoints-plus-longer/checkpoint-23500/
)
# /mnt/store/llama2-checkpoints-plus-freeze/checkpoint-46000/
count=0
# echo ${MODEL_LIST[@]}
while read server_ip; do
    # i=0
    # while [ $i -le 1 ]; do
        echo $server_ip
        # server_ip="10.24.116.75"
        # ssh -n $server_ip "PASSWORD=XMUMac2023"
        PASSWORD=XMUMac2023
        # ssh -n $server_ip "tmux new -s sbc_eval"
        tmux_session_name=sbc_eval
        ssh -n $server_ip "sudo -S apt install nfs-common <<< $PASSWORD"
        ssh -n $server_ip "sudo -S mount -t nfs 10.24.116.38:/data/nfs/share /mnt/SFT_store <<< $PASSWORD"
        ssh -n $server_ip "sudo -S mount -t nfs 10.24.116.39:/data/nfs/share /mnt/store <<< $PASSWORD"
        ssh -n $server_ip "tmux new -s sbc_eval"
        ssh -n $server_ip "tmux send-keys -t ${tmux_session_name} 'cd /mnt/SFT_store/3090_eval/FlagEvalMock_stable/' Enter"

        ssh -n $server_ip "tmux send-keys -t ${tmux_session_name} 'conda activate /mnt/SFT_store/xxw/envs/flageval' Enter"
        MODEL=llama2
        LLAMA_BASE=${MODEL_LIST[$count]}
        echo $LLAMA_BASE
        current_datetime=$(date +"%m%d_%H_%M_%S")
        mkdir -p "./logs/${MODEL}/${server_ip}/$(basename ${LLAMA_BASE})/"
        ssh -n $server_ip "tmux send-keys -t ${tmux_session_name} 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5' Enter"
        ssh -n $server_ip "tmux send-keys -t ${tmux_session_name} 'torchrun --nproc-per-node 6 main_v2dist.py --dataset-names="ALL" --model-name $MODEL --model-path $LLAMA_BASE --tokenizer-path $LLAMA_BASE  --batch-size 1 --verbose >> "./logs/${MODEL}/${server_ip}/$(basename ${LLAMA_BASE})/${current_datetime}.log" 2>&1' Enter"
        count=$((count+1))
        # i=$((i+1))
    # done
done < "./scripts/sbc_tmp/ssh_files.txt"