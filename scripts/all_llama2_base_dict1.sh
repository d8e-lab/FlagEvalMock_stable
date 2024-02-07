MODEL=InternLM
# directory=/mnt/SFT_store/xxw/outputs/0924_merge/
directory=$1

# 初始化一个空的列表
file_list=()

# 使用find命令查找当前目录下的目录文件，但排除指定目录本身
while IFS= read -r -d $'\0' file; do
  # 排除指定目录本身的路径
  if [ "$file" != "$directory" ]; then
    file_list+=("$file")
  fi
done < <(find "$directory" -maxdepth 1 -type d -print0)

mkdir -p ./logs/${MODEL}/$(basename "${LLAMA_BASE}")/
# 对文件目录下所有权重文件测评
for LLAMA_BASE in "${file_list[@]}"; do
{
    mkdir -p "./logs/${MODEL}/$(basename ${LLAMA_BASE})"
    echo ./logs/${MODEL}/$(basename "${LLAMA_BASE}")/
    current_datetime=$(date +"%m%d_%H_%M_%S")
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    torchrun --nproc-per-node 2 --master_port 25523 main_v2dist.py --dataset-names="ALL" \
        --model-name $MODEL \
        --model-path $LLAMA_BASE \
        --tokenizer-path $LLAMA_BASE \
        --batch-size 1 \
        --verbose \
        >> "./logs/${MODEL}/$(basename "${LLAMA_BASE}")/${current_datetime}.log" 2>&1
} done
