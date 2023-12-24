import os
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
from model_zoo import MODEL_DICT

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
    help="tokenizer路径，默认使用chat-glm",
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
ddp_setup()
gpu_id=int(int(os.environ["LOCAL_RANK"]))
# pdb.set_trace()
llm = model_builder(args.model_name, args.model_path, args.tokenizer_path, gpu_id=gpu_id)

if (
    os.path.exists(args.saver_path)
    and os.path.isfile(args.saver_path)
    and args.saver_path.endswith(".csv")
):
    result_df = pd.read_csv(args.saver_path)
else:
    result_df = pd.DataFrame(
        columns=["Model", "Use_Logits", "NT"]
        + ALL_DATASET
        + [i + "_time_cost" for i in ALL_DATASET]
    )

use_logits = [False] if not args.use_logits else [True]
# nts = [int(i) for i in args.nt.split(",")]
dataset_names = (
    ALL_DATASET if args.dataset_names == "ALL" else args.dataset_names.split(",")
)
dataset_names = [i.strip() for i in dataset_names if i.strip() in ALL_DATASET]
batch_size = args.batch_size

settings = list(itertools.product(use_logits, dataset_names))

with torch.no_grad():
    for use_logits, dataset_name in settings:
        nt=ALL_NEW_TOKENS[dataset_name]
        dataset_loader_time = time.time()
        dataset = get_dataset(dataset_name)
        dataset_loader_cost = time.time() - dataset_loader_time
        print(
            f"{dataset_name}  Loading Time Cost: {dataset_loader_cost}"
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,shuffle=False,sampler=DistributedSampler(dataset))
        print_rank0("dataset_name:", dataset_name)
        total_correct = 0
        total_len = 0
        # if use_logits and nt > 1:
        #     continue
        correct = 0
        choices = ALPHABET[:ALL_CLASSES[dataset_name]]
        tresponses = []
        tanswers = []
        start = time.time()
        outputs = []
        truth=[]
        # before_post=[]
        choiceses=[]
        for batch in tqdm(dataloader):
            queries = batch["prompt"]
            answers = batch["answer"]
            if hasattr(dataset, "generate_labels"):
                candidate = dataset.generate_labels(answers)
                if candidate is not None:
                    choiceses = candidate
            else:
                choiceses= [ choices for _ in queries]
            # results,bf = llm.inference(queries, choiceses, use_logits, nt, dataset_names)
            results = llm.inference(queries, choiceses, use_logits, nt, dataset_names)
            outputs.extend(results)
            truth.extend(answers)
            # before_post.extend(bf)
            if len(outputs) > 20:
                if args.verbose:
                    print_rank0(len(results), len(answers))
                    # print_rank0("before: ", before_post)
                    print_rank0("output: ", outputs)
                    print_rank0("truth : ", truth)
                outputs = []
                truth=[]
                before_post=[]
            correct += sum([1 if a.strip().strip(".") == i.strip() else 0 for (i, a) in zip(results, answers)])
        # exit()
        time_cost = time.time() - start
        correct_tensor=torch.tensor(correct,device=gpu_id)
        dist.barrier()
        dist.reduce(correct_tensor, dst=0, op=dist.ReduceOp.SUM)
        correct = int(correct_tensor)
        accuracy = correct / len(dataset)
        total_correct += correct
        total_len += len(dataset)
        acc = total_correct / total_len
        print_rank0(
            f"{dataset_name} {use_logits} {nt} Total Accuracy: {acc} Time Cost: {time_cost}"
        )
        condition = (
            (result_df["Model"] == model_name)
            & (result_df["Use_Logits"] == use_logits)
            & (result_df["NT"] == 1)
            # & (result_df["NT"] == nt)
        )
        if result_df.loc[condition].empty:
            new_row = [model_name, use_logits, nt] + [None] * 2 * len(ALL_DATASET)
            new_row[result_df.columns.get_loc(dataset_name)] = acc
            new_row[result_df.columns.get_loc(dataset_name + "_time_cost")] = time_cost
            result_df.loc[len(result_df)] = new_row
        else:
            result_df.loc[condition, dataset_name] = acc
            result_df.loc[condition, dataset_name + "_time_cost"] = time_cost
if not args.no_save:
    rank = dist.get_rank()
    if rank == 0:
        if (
            os.path.exists(args.saver_path)
            and os.path.isfile(args.saver_path)
            and args.saver_path.endswith(".csv")
        ):
            result_df.to_csv(args.saver_path, index=False)
        else:
            saved_name = args.saver_path
            if saved_name is None or saved_name == "":
                os.system("mkdir -p ./evaluation_results/")
                saved_name = "./evaluation_results/"
            saved_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            prefixs=args.model_path.split("/")
            if len(prefixs[-1]) > 0 and prefixs[-1] != "merge":
                filename_prefix=prefixs[-1]
            elif len(prefixs[-2]) > 0 and prefixs[-2] != "merge":
                filename_prefix=prefixs[-2]
            else:
                filename_prefix=args.model_name
            # filename_prefix=args.model_path.split("/")[-1] if len(args.model_path.split("/")[-1]) > 0 and args.model_path.split("/")[-1] != "merge" else args.model_path.split("/")[-2]
            saved_file = (
                # args.model_name
                filename_prefix
                + "_"
                + saved_time
                + "_"
                + "-".join(args.dataset_names.split(","))
                + "_results.csv"
            )
            saved_name = os.path.join(saved_name, saved_file)
            result_df.to_csv(saved_name, index=False)
print_rank0("Evaluation Total Time Cost:", time.time() - all_start)

