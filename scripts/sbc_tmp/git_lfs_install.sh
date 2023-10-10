ssh_files="./scripts/sbc_tmp/ssh_files.txt"
PASSWORD="XMUMac2023"
url_list=(
    "https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat"
    "https://huggingface.co/baichuan-inc/Baichuan2-13B-Base"
    "https://huggingface.co/baichuan-inc/Baichuan-13B-Chat"
    "https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat"
    "https://huggingface.co/baichuan-inc/Baichuan2-7B-Base"
    "https://huggingface.co/Qwen/Qwen-7B"
)
# while read -r url; do
#     url_list+=("$url")
#     echo $url
# done < "./scripts/sbc_tmp/model_url.txt"
count=0
tmux_session_name=sbc_eval
echo ${url_list[@]}
while read server_ip; do
    echo $server_ip
    # ssh -n $server_ip "curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo -S bash <<< $PASSWORD"
    # ssh -n $server_ip "sudo -S apt-get install git-lfs <<< $PASSWORD"
    ssh -n $server_ip "tmux send-keys -t ${tmux_session_name} 'export https_proxy=http://127.0.0.1:15777 http_proxy=http://127.0.0.1:15777' Enter"
    ssh -n $server_ip "tmux send-keys -t ${tmux_session_name} 'git config --global http.proxy http://127.0.0.1:15777 && git config --global https.proxy http://127.0.0.1:15777' Enter"
    ssh -n $server_ip "tmux send-keys -t ${tmux_session_name} 'cd /mnt/store' Enter"
    ssh -n $server_ip "tmux send-keys -t ${tmux_session_name} 'git lfs install' Enter"
    ssh -n $server_ip "tmux send-keys -t ${tmux_session_name} 'git clone ${url_list[$count]} > /mnt/SFT_store/3090_eval/FlagEvalMock_stable/logs/${server_ip}.log 2>&1' Enter"
    count=$((count+1))
done < $ssh_files