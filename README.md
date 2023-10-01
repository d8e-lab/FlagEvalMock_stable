
# Model Evaluation Script

This script provides a way to evaluate pre-trained models on different datasets. It supports various configurations and can be used to understand the performance and time cost associated with different models.

## Requirements

- Python 3.x
- Torch
- Pandas
- tqdm

## Description

The script accepts a range of parameters for defining the model to be evaluated, the path to the model, tokenizer, datasets, and other specific settings. Results are saved as a CSV file that records the model's accuracy and time cost on different datasets.

## Features

- Supports multiple models as defined in `MODEL_DICT`.
- Can evaluate the model on various datasets defined in `ALL_DATASET`.
- Can save the evaluation results in a CSV file.
- The script provides verbose output for detailed debugging and analysis.

## Usage

The following command-line arguments are supported:

- `--model-name`: The name of the model to be used. Default is "llama2."
- `--model-path`: The path to the model. Default is "/data/LLM/Llama-2-7b-hf/".
- `--tokenizer-path`: The path to the tokenizer. Default is "/data/LLM/Llama-2-7b-hf/".
- `--dataset-names`: The dataset names to be used. Default is "ALL."
- `--saver-path`: The path to save the evaluation results. Default is an empty string, which will create a new CSV file.
- `--use-logits`: A flag to determine if logits should be used.
- `--batch-size`: Batch size for evaluation. Default is 4.
- `--no-save`: A flag to prevent saving the results.
- `--verbose`: A flag for verbose output.

Example usage:

```bash
python script_name.py --model-name llama2 --dataset-names "Dataset1, Dataset2" --batch-size 8
```

You can use the following shell command to automatically evaluate multiple model paths in the specified directory using multiple GPUs:
```shell
MODEL=<Your Model Name>
directory=/path/to/direction

# Initialize an empty list
file_list=()

# Use the find command to search for directory files in the current directory, excluding the specified directory itself
while IFS= read -r -d $'\0' file; do
  # Exclude the path of the specified directory itself
  if [ "$file" != "$directory" ]; then
    file_list+=("$file")
  fi
done < <(find "$directory" -maxdepth 1 -type d -print0)

mkdir -p "./logs/${MODEL}/$(basename ${LLAMA_BASE})/"

# Evaluate all model files in the directory
for LLAMA_BASE in "${file_list[@]}"; do
{
    mkdir -p "./logs/${MODEL}/$(basename ${LLAMA_BASE})"
    current_datetime=$(date +"%m%d_%H_%M_%S")
    CUDA_VISIBLE_DEVICES=1,2,3,4,6,7 torchrun --nproc-per-node 6 main_v2dist.py --dataset-names="ALL" \
        --model-name $MODEL \
        --model-path $LLAMA_BASE \
        --tokenizer-path $LLAMA_BASE \
        --batch-size 1 \
        --verbose \
        >> "./logs/${MODEL}/$(basename ${LLAMA_BASE})/${current_datetime}.log" 2>&1
} done
```
The purpose of this script is to evaluate all model files in the specified directory. It performs model evaluation for each model file and stores the results in log files. Make sure to define the MODEL and directory variables before running the script and modify their values according to your requirements. This script sequentially evaluates the model files listed in the specified directory and stores the results in log files.  
Please note that if your model path is "/data/LLM/Llama-2-7b-hf/", your directory variable should be set to "/data/LLM/", and this directory should not contain any non-model directories.


## Output

The script will print the dataset name, usage of logits, total accuracy, and time cost. It will also save the results in a CSV file if the `--no-save` option is not used.

## Notes

- Ensure that the models and datasets are correctly defined in `MODEL_DICT`, `ALL_DATASET`, and other related variables.
- The paths for the model and tokenizer should be correctly specified.
- Some parts of the code are commented out and might be related to specific use cases or further development.
