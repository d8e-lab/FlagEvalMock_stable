MODEL=llama2
directory="/mnt/store/llama2-checkpoints-plus-sft-v3/"

# 初始化一个空的列表
file_list=()

# 使用find命令查找当前目录下的目录文件，但排除指定目录本身
while IFS= read -r -d $'\0' file; do
  # 获取目录名称中的数字部分
  dir_name=$(basename "$file")
  dir_number="${dir_name#checkpoint-}"

  # 检查数字是否大于7250
  if [[ "$dir_number" -gt 11750 ]]; then
    file_list+=("$file")
  fi
done < <(find "$directory" -maxdepth 1 -type d -print0)

for LLAMA_BASE in "${file_list[@]}"; do
{   
    echo $LLAMA_BASE
    mkdir -p "./logs/${MODEL}/$(basename ${LLAMA_BASE})"
    current_datetime=$(date +"%m%d_%H_%M_%S")
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
    torchrun --nproc-per-node 6 main_v2dist.py --dataset-names="ALL" \
        --model-name $MODEL \
        --model-path $LLAMA_BASE \
        --tokenizer-path $LLAMA_BASE \
        --batch-size 1 \
        --verbose \
        >> "./logs/${MODEL}/$(basename ${LLAMA_BASE})/${current_datetime}.log" 2>&1
} done