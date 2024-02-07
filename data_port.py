import os
import sys
import pdb
from tqdm import tqdm

from flageval_datasets import (
    CEvalDataset,
    LinkSoulCEvalDataset,
    BUSTMDataset,
    OCNLIDataset,
    GAOKAO2023Dataset,
    TruthfulQADataset,
    EPRSTMTDataset,
    TNEWSDataset,
    CMMLUDataset,
    ChIDDataset,
    CSLDataset,
    CLUEWSCDataset,
    RAFTDataset,
    TruthfulQADataset,
    IMDBDataset,
    BoolQDataset,
    MMLUDataset,
    NewMMLUDataset,
    huggingface_datasets,
)
import torch
from torch.utils.data import ConcatDataset

ALL_DATASET = [
    "BoolQ",
    "MMLU",
    "TruthfulQA",
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
    "IMDB",
]
ALL_DIRPATH_DIC = {
    "BUSTM": "/mnt/SFT_store/flagevalmock/BUSTM",
    "OCNLI": "/mnt/SFT_store/flagevalmock/OCNLI",
    "EPRSTMT": "/mnt/SFT_store/flagevalmock/EPRSTMT",
    "TNEWS": "/mnt/SFT_store/flagevalmock/TNEWS",
    "ChID": "/mnt/SFT_store/flagevalmock/chid",
    "CSL": "/mnt/SFT_store/flagevalmock/csl",
    "CLUEWSC": "/mnt/SFT_store/flagevalmock/cluewsc",
}
ALL_PATH_DIC = {
    "C-Eval": "/mnt/SFT_store/flagevalmock/C-Eval/C-Eval-preprocess/val",
    "GAOKAO2023": "/mnt/SFT_store/flagevalmock/GAOKAO2023",
    "Chinese_MMLU": "/mnt/SFT_store/flagevalmock/cmmlu/dev",  # test 无标签
}

CLASSES = 24
ALL_CLASSES = {
    "BoolQ": 2,
    "MMLU": 4,
    "TruthfulQA": 4,
    "IMDB": 2,
    "RAFT": 26, # 多个子集，选项数目不定
    "Chinese_MMLU":4,
    "C-Eval":4,
    "GAOKAO2023":4,
    "CSL":2,
    "ChID":7,
    "CLUEWSC": 2, #2, # 形式很奇怪，看看咋回事
    "EPRSTMT":2,
    "TNEWS":15,
    "OCNLI":3, # Neural, Entailment, Contradiction
    "BUSTM":2,

}
ALL_NEW_TOKENS = {
    "BoolQ": 1,
    "MMLU": 1,
    "TruthfulQA": 1,
    "IMDB": 4,
    "RAFT": 16, # 多个子集，选项数目不定
    "Chinese_MMLU":1,
    "C-Eval":1,
    "GAOKAO2023":1,
    "CSL":1, # 1, # 形式很奇怪，看看咋回事
    "ChID":1,
    "CLUEWSC":1, # 1, # 形式很奇怪，看看咋回事
    "EPRSTMT":1,
    "TNEWS":1,
    "OCNLI":1, # Neural, Entailment, Contradiction
    "BUSTM":1,

}
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def get_dataset(dataset_name: str = ""):
    assert dataset_name in ALL_DATASET
    if dataset_name in huggingface_datasets:
        if dataset_name == "RAFT":
            dataset = RAFTDataset()
        elif dataset_name == "TruthfulQA":
            dataset = TruthfulQADataset()
        elif dataset_name == "IMDB":
            dataset = IMDBDataset()
        elif dataset_name == "BoolQ":
            dataset = BoolQDataset()
        elif dataset_name == "MMLU":
            # dataset = MMLUDataset()
            dataset = NewMMLUDataset() # /mnt/SFT_store/3090_eval/FlagEvalMock_stable/wyh_sample/23_no_imdb.sh
        elif dataset_name == "Chinese_MMLU":
            dataset = CMMLUDataset()
    elif dataset_name in ALL_DIRPATH_DIC:
        datasets = []
        valjson = ALL_DIRPATH_DIC[dataset_name]
        try:
            if dataset_name == "BUSTM":
                dataset = BUSTMDataset(valjson)
            elif dataset_name == "OCNLI":
                dataset = OCNLIDataset(valjson)
            elif dataset_name == "EPRSTMT":
                dataset = EPRSTMTDataset(valjson)
            elif dataset_name == "TNEWS":
                dataset = TNEWSDataset(valjson)
            elif dataset_name == "ChID":
                dataset = ChIDDataset(valjson)
            elif dataset_name == "CSL":
                dataset = CSLDataset(valjson)
            elif dataset_name == "CLUEWSC":
                dataset = CLUEWSCDataset(valjson)
            assert len(dataset) > 0
        except Exception as e:
            print(valjson)
            print(e)
            exit()
        datasets.append(dataset)
    else:
        datasets = []
        for root, dirs, files in os.walk(ALL_PATH_DIC[dataset_name]):
            for file in tqdm(files):
                valjson = os.path.join(root, file)
                try:
                    if dataset_name == "C-Eval":
                        dataset = CEvalDataset(valjson)
                        # dataset = LinkSoulCEvalDataset(valjson)
                    elif dataset_name == "GAOKAO2023":
                        dataset = GAOKAO2023Dataset(valjson)
                    # elif dataset_name == "Chinese_MMLU":
                    #     dataset = CMMLUDataset(valjson)
                    if len(dataset) == 0:
                        continue
                except Exception as e:
                    print(valjson)
                    print(e)
                    exit()
                datasets.append(dataset)
        dataset = ConcatDataset(datasets=datasets)
    return dataset

