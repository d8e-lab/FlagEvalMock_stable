import os
import json
import random
import copy
import torch
import datasets
import pdb
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class RAFTDataset(Dataset):
    """
    等待施工。。。
    """

    def __init__(self, ceval_path="", subset_name="", using_gpt=False, item_size=5):
        subset_name = [
            "ade_corpus_v2",
            "banking_77",
            "neurips_impact_statement_risks",
            "one_stop_english",
            "overruling",
            "semiconductor_org_types",
            "systematic_review_inclusion",
            "tai_safety_research",
            "terms_of_service",
            "tweet_eval_hate",
            "twitter_complaints",
        ]
        self.first_line = "The following content is text labeling mission about "
        self.item_size = item_size

        self.sub2ind = {}
        self.sub2label = {}
        self.dataset = []
        self.train_dataset[]
        i = 0
        for sub in subset_name:
            d = datasets.load_dataset("ought/raft", sub)
            # pdb.set_trace()
            # train集和test集的label应该是一样的，感觉不需要改
            lable = d["train"].features["Label"]
            self.sub2label[sub] = lable
            # for split in d:
            for item in d["train"]:
                if item["Label"] == 0:
                    continue
                self.train_dataset.append(item)
                self.train_subname2index.setdefault(sub,[]).append(i)
                i += 1
            # 原版的循环应该先访问train集，序号上应该会对的上，这个是个潜在的bug？
            for item in d["test"]:
                if item["Label"] == 0:
                    continue  # skip unlabeled
                self.dataset.append(item)
                # 题号绑定子集名
                self.sub2ind.setdefault(sub, []).append(i)
                i += 1
    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        # Select random data samples from the dataset
        sub_name = ""
        for n, inds in self.train_subname2index.items(): 
            # n 即 subset_name，ban_index不懂是什么
            if ban_index in inds:
                sub_name = n
                break
        pdb.set_trace()
        # ban_ind->sub_name->定题型?
        prompt = self.first_line+sub_name+".\n"
        # 根据subset_name获取文本分类的答案标签，如【1.ADE-related 2.not ADE-related】,
        # 如果修改需要单独的sub2ind，因为subset_name和题目id绑定，根据ban_index选定subset_name，
        # 如果不修改可能ban_index会指向错误subset_name
        labels = self.sub2label[sub_name].names # dataset['train'].features['Label'].names
        prompt_possible_answers = [f"{i}. {labels[i]}\n" for i in range(1, len(labels))]
        prompt += "".join(prompt_possible_answers) + "\n"
        # sub2ind[sub_name]
        inds = random.sample(self.train_subname2index[sub_name], 5)
        for i in inds:
            item = self.train_dataset[i]
            item_prompt = ""
            # item格式：{...., 'ID': 28, 'Label': 2}
            for k, v in item.items():
                # if k in ["ID", "id", "Id", "iD"]:
                if k.lower() == "id":
                    continue
                if k == "Label":
                    continue
                item_prompt += f"{k}: {v}\n"
            item_prompt += f"Label : {labels[item['Label']]}\n"
            prompt += item_prompt + "\n"
        return prompt, labels