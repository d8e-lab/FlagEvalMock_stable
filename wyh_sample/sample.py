# import os
import math
# import glob
import random
# import re
# import json
import time
import numpy as np
import torch
import torch.utils.data
# from tqdm import tqdm
# import pandas as pd
# import itertools
# import pdb

# import argparse
from data_port import get_dataset, ALL_DATASET
# from model_zoo import MODEL_DICT

# from linly_llm import LinlyLLaMA2
# from functools import partial

#parallel
# import torch.distributed as dist
# from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Subset,ConcatDataset
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group
# def ddp_setup():
#     init_process_group(backend="nccl")

# def print_rank0(*args, **kwargs):
#     if dist.is_initialized():
#         rank = dist.get_rank()
#         if rank == 0:
#             print(*args, **kwargs)
#     else:
#         print(*args, **kwargs)
        
def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42) 
# all_start = time.time()
# parser = argparse.ArgumentParser(description="SETTINGS for MODELS")
# parser.add_argument(
#     "--model-name", type=str, default="llama2", help="模型名称，默认使用chat-glm"
# )
# parser.add_argument(
#     "--model-path", type=str, default="/data/LLM/Llama-2-7b-hf/", help="模型路径，默认使用chat-glm"
# )
# parser.add_argument(
#     "--tokenizer-path",
#     type=str,
#     default="/data/LLM/Llama-2-7b-hf/",
#     help="tokenizer路径，默认使用chat-glm",
# )

# parser.add_argument("--dataset-names", type=str, default="ALL", help="数据集名称")

# parser.add_argument("--saver-path", type=str, default="", help="保存路径")
# parser.add_argument("--use-logits", action="store_true", help="")
# # parser.add_argument("--nt", type=str, default="1", help="nt")
# parser.add_argument("--batch-size", type=int, default=4, help="batch size")
# parser.add_argument("--no-save", action="store_true", help="set to not save")
# parser.add_argument("--verbose", action="store_true", help="print answers")
# parser.add_argument(
#     "--ratio", type=float, default="0.1", help="采样比例"
# )
# args = parser.parse_args()
ALL_DATASET = [
    "BoolQ",
    "MMLU",
    "TruthfulQA",
    "IMDB",
    "RAFT",
    "Chinese_MMLU",
    "C-Eval",
    "GAOKAO2023",
    "CSL",
    "ChID",
    "CLUEWSC",
    "EPRSTMT",
    "TNEWS",
    "OCNLI",
    "BUSTM",
]
ALL_RATIO = {
    "BoolQ": 0.118 ,
    "MMLU": 0.07,
    "TruthfulQA": 0.05,
    "IMDB": 0.005,
    "RAFT": 0.05, # 多个子集，选项数目不定
    "Chinese_MMLU":0.05,
    "C-Eval":0.05,
    "GAOKAO2023":0.07,
    "CSL":0.07, # 1, # 形式很奇怪，看看咋回事
    "ChID":0.07,
    "CLUEWSC":0.05, # 1, # 形式很奇怪，看看咋回事
    "EPRSTMT":0.01,
    "TNEWS":0.05,
    "OCNLI":0.08, # Neural, Entailment, Contradiction
    "BUSTM":0.18,
}


def GetAuxiliaryDataset():
    dataset_names = ALL_DATASET
    total=0
    dataset_list=[]
    with torch.no_grad():
        for dataset_name in dataset_names:
            dataset_loader_time = time.time()
            dataset = get_dataset(dataset_name)
            dataset_loader_cost = time.time() - dataset_loader_time
            # print(
            #     f"{dataset_name}  Loading Time Cost: {dataset_loader_cost}"
            # )
            ind=math.ceil(ALL_RATIO[dataset_name]*len(dataset))
            # print(f"{dataset_name} full length is : {len(dataset)}")
            dataset=Subset(dataset=dataset,indices=range(ind))
            # print(f"{dataset_name} sample length is : {len(dataset)}")
            total+=len(dataset)
            dataset_list.append(dataset)
    dataset = ConcatDataset(datasets=dataset_list)
    return dataset

dataset=GetAuxiliaryDataset()
print(f"Finally, total Auxiliary Training Set length is {len(dataset)}")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2,shuffle=False)
