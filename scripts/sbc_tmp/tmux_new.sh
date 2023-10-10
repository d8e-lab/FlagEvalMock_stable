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
        ssh -n $server_ip "sudo -S mkdir -p /mnt/40_store <<< $PASSWORD"
        ssh -n $server_ip "sudo -S chmod 777 /mnt/40_store <<< $PASSWORD"
        ssh -n $server_ip "sudo -S mount -t nfs 10.24.116.40:/data/nfs/share /mnt/40_store <<< $PASSWORD"
        # ssh -t $server_ip "tmux new -s sbc_eval"
done < "./scripts/sbc_tmp/ssh_files.txt"