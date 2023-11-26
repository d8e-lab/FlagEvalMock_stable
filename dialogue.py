import os
import sys
import glob
import random
import re
import json
import time
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
import pandas as pd
import itertools
import pdb

import argparse
from data_port import get_dataset, ALL_DATASET, ALPHABET, ALL_CLASSES, ALL_NEW_TOKENS
from model_zoo import MODEL_DICT,postprocess,preprocess

# from linly_llm import LinlyLLaMA2
from functools import partial

#parallel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
def ddp_setup():
    init_process_group(backend="nccl")

def print_rank0(*args, **kwargs):
    if dist.is_initialized():
        rank = dist.get_rank()
        if rank == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)
        
def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42) 
all_start = time.time()
parser = argparse.ArgumentParser(description="SETTINGS for MODELS")
parser.add_argument(
    "--model-name", type=str, default="llama2", help="模型名称，默认使用chat-glm"
)
parser.add_argument(
    "--model-path", type=str, default="/data/LLM/Llama-2-7b-hf/", help="模型路径，默认使用chat-glm"
)
parser.add_argument(
    "--tokenizer-path",
    type=str,
    default="/data/LLM/Llama-2-7b-hf/",
    help="tokenizer路径",
)

parser.add_argument("--dataset-names", type=str, default="ALL", help="数据集名称")

parser.add_argument("--saver-path", type=str, default="", help="保存路径")
parser.add_argument("--use-logits", action="store_true", help="")
# parser.add_argument("--nt", type=str, default="1", help="nt")
parser.add_argument("--batch-size", type=int, default=4, help="batch size")
parser.add_argument("--no-save", action="store_true", help="set to not save")
parser.add_argument("--verbose", action="store_true", help="print answers")
args = parser.parse_args()

model_name = args.model_name
model_builder = MODEL_DICT[args.model_name]
# model_builder=ChatGLM2
# 模型加载
# gpu_id=int(int(os.environ["LOCAL_RANK"]))
llm = model_builder(args.model_name, args.model_path, args.tokenizer_path, gpu_id=0)

# Define the exit key
exit_key = "exit()"
dataset_names = ALL_DATASET
dataloaders={}
answers=[]
for dataset_name in dataset_names:
    dataset = get_dataset(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True)
    dataloaders[dataset_name]=dataloader

# for k,d in dataloaders.items():
#     print(k)
#     for batch in d:
#         queries = batch["prompt"]
#         answers = batch["answer"]
#         llm.chat(queries, [[""]], False, 128, "Dialogue")

# Start the dialogue loop
with torch.no_grad():
    while True:
        # Get user input
        user_input = input("You: ")
        # Check if the user wants to exit
        if user_input.lower() == exit_key:
            print("Exiting the dialogue loop.")
            break
        print("INPUT: ",user_input)
        user_input=preprocess(user_input)
        print("AFTER: ",user_input)
        results = llm.chat([user_input], [[""]], False, 128, "Dialogue")
        # Use your LLM inference function to generate a response
        # Print the LLM's response
        print("RESULT: ",results[0])
        results=[ postprocess(i) for i in results]
        print("PROCED: ",results[0])

