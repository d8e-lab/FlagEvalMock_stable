import json
import random
import copy
import torch
import datasets
import pdb
import os
import threading
import re
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
B_INST, E_INST = "[INST]", "[/INST]"

huggingface_datasets = ["RAFT", "TruthfulQA", "IMDB", "BoolQ", "MMLU"]


class CEvalDataset(Dataset):
    def __init__(self, ceval_path, using_gpt=False, item_size=5):
        directory = os.path.dirname(ceval_path)
        dev_directory = directory.replace('/val','/dev')
        root,dirs,files= next(iter(os.walk(dev_directory)))
        file = next(iter(files))
        file = os.path.join(dev_directory,file)
        with open(file,"r",encoding="utf-8") as file:
            self.dev_dataset = json.load(file)
        with open(ceval_path, "r", encoding="utf-8") as file:
            self.dataset = json.load(file)
        if len(ceval_path.split("\\")) > 1:
            subject_name = " ".join(ceval_path.split("\\")[-1][:-4].split("_")[:-1])
        else:
            subject_name = " ".join(ceval_path.split("/")[-1][:-4].split("_")[:-1])
        self.name = "CEval/" + subject_name
        # self.name=subject_name
        self.first_line = (
            # "这个任务是中国关于"
            "以下是关于"
            + subject_name
            + "的选择题"
            # + "考试的问题，请从给出的A、B、C、D四个选项中，选出其中的正确答案。请回答'A'或'B'或'C'或'D'\n"
        )
        if using_gpt:  # 是否使用gpt的提示作为first line
            prompts = [
                "请回答下列问题。",
                "在每个问题中选择正确的答案。",
                "请仔细阅读问题并选择最合适的选项。",
                "在每个问题中选择最恰当的答案。",
                "请从给定的选项中选择正确的答案以回答问题。",
                "根据问题要求选择最佳答案。",
                "在每个问题的选项中，找出与问题描述最匹配的答案。",
                "请根据问题描述，从提供的选项中选择正确的答案。",
                "在每个问题中，选择最适合的选项以回答问题。",
                "根据问题描述，从给定的选项中选择最合适的答案。",
                "请从提供的选项中选择最适合问题描述的答案。",
                "根据问题描述，选择下列选项中最准确的答案。",
                "在每个问题中，从给定的选项中选择最符合要求的答案。",
                "请仔细阅读每个问题，并从给定的选项中选择最适合的答案。",
                "根据问题描述，在每个问题中选择最佳答案以回答问题。",
            ]
            idx = random.sample(range(0, 15), 1)[0]
            self.first_line = prompts[idx]
        self.item_size = item_size

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        # json_data = self.dataset
        json_data = self.dev_dataset
        idns = random.sample(range(0, len(json_data)), self.item_size)
        max_try = 10
        while ban_index in idns and max_try > 0:
            idns = random.sample(range(0, len(json_data)), self.item_size)
            max_try -= 1
        if max_try == 0:
            print("Warning: cannot generate prompt without Question index")
        prompt = self.first_line
        for idx in idns:
            entry = json_data[idx]
            question = entry["question"]
            choices = entry["choices"]
            answer = entry["answer"]
            # answer = entry["answer"]+'.'+choices[ord(entry["answer"])-65]

            formatted_string = f"请效仿此示例：问题:{question}\n"
            formatted_string += "选项："
            formatted_string += "\n".join(
                [f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]
            )
            formatted_string += f"\n答案: {answer}"

            prompt = prompt + "\n\n" + formatted_string
        # print("prompt：",prompt)
        return prompt

    def __getitem__(self, index):
        # prompt = self.first_line
        # if torch.is_tensor(index):
        #     index = index.tolist()
        # if index is iterable:
        # if not isinstance(index, list):
        #     index = [index]
        # print(type(index),index)
        sample = []
        idx = index
        # for idx in index:
        prompt = self.__generate_prompt__(idx)
        # prompt = self.first_line
        entry = self.dataset[idx]
        question = entry["question"]
        choices = entry["choices"]
        answer = entry["answer"]
        # answer = entry["answer"]+'.'+choices[ord(entry["answer"])-65]

        formatted_string = f"问题:{question}\n"
        formatted_string += "选项：\n"
        formatted_string += "\n".join(
            [f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]
        )
        formatted_string += f"\n答案: "
        prompt = prompt + "\n\n" + formatted_string
        sample = {"prompt": prompt, "answer": answer}
        # sample.append([prompt, answer])
        return sample
        # else:
        # prompt = self.__generate_prompt__(index)
        # entry = self.dataset[index]
        # question = entry['question']
        # choices = entry['choices']
        # answer = entry['answer']

        # formatted_string = f"问题:{question}\n"
        # formatted_string += '\n'.join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        # formatted_string += f"\n答案: "
        # prompt = prompt + "\n\n" + formatted_string
        # sample = [prompt, answer]
        # return [sample]

class LinkSoulCEvalDataset(CEvalDataset):
    def __init__(self, ceval_path, using_gpt=False, item_size=5):
        super().__init__(ceval_path, using_gpt, item_size)
        self.first_line = B_SYS + self.first_line + E_SYS


    def __getitem__(self, index):
        # prompt = self.first_line
        # if torch.is_tensor(index):
        #     index = index.tolist()
        # if index is iterable:
        # if not isinstance(index, list):
        #     index = [index]
        # print(type(index),index)
        sample = []
        idx = index
        # for idx in index:
        prompt = self.__generate_prompt__(idx)
        # prompt = self.first_line
        entry = self.dataset[idx]
        question = entry["question"]
        choices = entry["choices"]
        answer = entry["answer"]
        # answer = entry["answer"]+'.'+choices[ord(entry["answer"])-65]

        formatted_string = f"问题:{question}\n"
        formatted_string += "选项：\n"
        formatted_string += "\n".join(
            [f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]
        )
        formatted_string += f"\n答案: "
        prompt = prompt + "\n\n" + formatted_string
        sample = {"prompt": f"{B_INST} {prompt} {E_INST} ", "answer": answer}
        # sample.append([prompt, answer])
        return sample
    

class BUSTMDataset(Dataset):
    def __init__(self, ceval_path, using_gpt=False, item_size=5):
        
        dev_path = os.path.join(ceval_path,'dev_few_all.json')
        test_path = os.path.join(ceval_path,'test_public.json')
        # train集不读取
        with open(dev_path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.readlines()
            self.dev_dataset = list(map(lambda x: json.loads(x), data))

        with open(test_path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.readlines()
            self.dataset = list(map(lambda x: json.loads(x), data))
        self.name = "BUSTM"
        # self.first_line = "请根据提供的中文句子，判断它们是否属于同一语义：\n"
        self.first_line = "文本语义是否相似\n"
        self.item_size = item_size
        self.prompt_dict = {"1": "A", "0": "B"}

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        # Select random data samples from the dataset
        samples = random.sample(self.dev_dataset, self.item_size)
        # Initialize the prompt string
        prompt = "请根据提供的中文句子，判断它们是否属于同一语义：\n"

        for i, sample in enumerate(samples):
            sentence1 = sample["sentence1"]
            sentence2 = sample["sentence2"]
            label = sample["label"]
            # Add the sample information to the prompt
            prompt += f"\n文本:"
            prompt += f"\n文本1: {sentence1}"
            prompt += f"\n文本2: {sentence2}"
            prompt += f"\nA. 相似"
            prompt += f"\nB. 不相似"
            prompt += f"\n答案: {self.prompt_dict[label]}\n"
        return prompt

    def __getitem__(self, index):
        sample = []
        idx = index
        prompt = self.__generate_prompt__(idx)
        entry = self.dataset[idx]
        sentence1 = entry["sentence1"]
        sentence2 = entry["sentence2"]
        answer = entry["label"]
        prompt += f"\n文本:"
        prompt += f"\n文本1: {sentence1}"
        prompt += f"\n文本2: {sentence2}"
        prompt += f"\nA. 相似"
        prompt += f"\nB. 不相似"
        prompt += f"\n答案: \n"
        sample = {"prompt": prompt, "answer": self.prompt_dict[answer]}
        return sample


class OCNLIDataset(Dataset):
    def __init__(self, ceval_path, using_gpt=False, item_size=5):
        # ceval_path: ./OCNLI文件夹路径
        dev_path = os.path.join(ceval_path,'dev_few_all.json')
        test_path = os.path.join(ceval_path,'test_public.json')
        # train集不读取
        with open(dev_path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.readlines()
            self.dev_dataset = list(map(lambda x: json.loads(x), data))

        with open(test_path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.readlines()
            self.dataset = list(map(lambda x: json.loads(x), data))
        self.name = "OCNLI"
        self.first_line = "推理下列两句话的关系\n"
        self.item_size = item_size

        for data in self.dataset.copy():
            if data["label"] == "-":
                self.dataset.remove(data)

        self.prompt_dict = {"contradiction": "A", "neutral": "B", "entailment": "C"}

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        # Select random data samples from the dataset
        samples = random.sample(self.dev_dataset, self.item_size)
        # Initialize the prompt string
        prompt = self.first_line

        for i, sample in enumerate(samples):
            sentence1 = sample["sentence1"]
            sentence2 = sample["sentence2"]
            label = sample["label"]

            # Add the sample information to the prompt
            prompt += f"\n文本:"
            prompt += f"\n文本1: {sentence1}"
            prompt += f"\n文本2: {sentence2}"
            prompt += f"\nA: 冲突"
            prompt += f"\nB: 中立"
            prompt += f"\nC: 蕴含"
            prompt += f"\n答案: {self.prompt_dict[label]}\n"
        return prompt

    def __getitem__(self, index):
        sample = []
        idx = index
        prompt = self.__generate_prompt__(idx)
        entry = self.dataset[idx]
        sentence1 = entry["sentence1"]
        sentence2 = entry["sentence2"]
        answer = entry["label"]
        prompt += f"\n文本:"
        prompt += f"\n文本1: {sentence1}"
        prompt += f"\n文本2: {sentence2}"
        prompt += f"\nA: 冲突"
        prompt += f"\nB: 中立"
        prompt += f"\nC: 蕴含"
        prompt += f"\n答案: \n"
        sample = {"prompt": prompt, "answer": self.prompt_dict[answer]}
        return sample


class GAOKAO2023Dataset(Dataset):# 只有测试集，不做修改
    def __init__(self, ceval_path, using_gpt=False, item_size=5):
        self.dataset = json.load(open(ceval_path, "r", encoding="utf-8"))

        self.name = "GAOKAO2023"
        file_name = os.path.basename(ceval_path)
        subject = file_name.split('_')[1]
        self.first_line = "以下是2023 年高考的选择题（附答案），请从四个选项里选择正确答案.\n"
        self.item_size = item_size

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        # Select random data samples from the dataset
        samples = random.sample(self.dataset, self.item_size)
        # Initialize the prompt string
        prompt = self.first_line

        for i, sample in enumerate(samples):
            # Add the sample information to the prompt
            question = re.sub(r'[0-9]+．[ ]*（[ ]*[0-9]+分[ ]*）[ ]*','',str(sample["question"]))
            question = re.sub(r'[0-9]+.[ ]*\([ ]*[0-9]+[ ]+分[ ]*\)[ ]*','',question)
            question = re.sub(r'^[0-9]+[ ]*．[ ]*','',question)
            question = re.sub(r'^[0-9]+[ ]*.[ ]*','',question)
            question = re.sub(r'(?<![.\?!:"\'…。（）\)])\n', '', question)
            prompt += "题目: " + question + "\n"
            prompt += str(sample["choices"])
            prompt += "答案: " + str(sample["answer"][0]) + "\n"
            prompt += "\n"

        return prompt

    def __getitem__(self, index):
        sample = []
        idx = index
        prompt = self.__generate_prompt__(idx)
        entry = self.dataset[idx]
        answer = entry["answer"][0]
        question = re.sub(r'[0-9]+．[ ]*（[ ]*[0-9]+分[ ]*）[ ]*','',str(entry["question"]))
        question = re.sub(r'[0-9]+.[ ]*\([ ]*[0-9]+[ ]+分[ ]*\)[ ]*','',question)
        question = re.sub(r'^[0-9]+[ ]*．[ ]*','',question)
        question = re.sub(r'^[0-9]+[ ]*.[ ]*','',question)
        question = re.sub(r'(?<![.\?!:"\'…。（）\)])\n', '', question)
        prompt += "题目: " + question + "\n"
        prompt += str(entry["choices"]) + "\n"
        prompt += "答案: " + "\n"
        prompt += "\n"

        sample = {"prompt": prompt, "answer": answer}
        return sample


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
        i = 0
        for sub in subset_name:
            d = datasets.load_dataset("ought/raft", sub)
            lb = d["train"].features["Label"]
            self.sub2label[sub] = lb
            for split in d:
                for item in d[split]:
                    if item["Label"] == 0:
                        continue  # skip unlabeled
                    self.dataset.append(item)
                    # 题号绑定子集名
                    # 可以优化，这里可以只保存题型中最大最小题号，后续查找检查在哪一个题型的题号范围内即可
                    self.sub2ind.setdefault(sub, []).append(i)
                    i += 1

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        # Select random data samples from the dataset
        sub_name = ""
        for n, inds in self.sub2ind.items(): 
            # n 即 subset_name，ban_index不懂是什么
            if ban_index in inds:
                sub_name = n
                break
        # ban_ind->sub_name->定题型?
        prompt = self.first_line+sub_name+".\n"
        # 根据subset_name获取文本分类的答案标签，如【1.ADE-related 2.not ADE-related】,
        # 如果修改需要单独的sub2ind，因为subset_name和题目id绑定，根据ban_index选定subset_name，
        # 如果不修改可能ban_index会指向错误subset_name
        labels = self.sub2label[sub_name].names # dataset['train'].features['Label'].names
        prompt_possible_answers = [f"{i}. {labels[i]}\n" for i in range(1, len(labels))]
        prompt += "".join(prompt_possible_answers) + "\n"
        # sub2ind[sub_name]
        inds = random.sample(self.sub2ind[sub_name], 5)
        for i in inds:
            item = self.dataset[i]
            item_prompt = ""
            # item格式：{...., 'ID': 28, 'Label': 2},可以确认一下i其实等于ID
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

    def __getitem__(self, index):
        prompt, labels = self.__generate_prompt__(index)
        item = self.dataset[index]
        item_prompt = ""
        for k, v in item.items():
            if k in ["ID", "id", "Id", "iD"]:
                continue
            if k == "Label":
                continue
            item_prompt += f"{k}: {v}\n"
        item_prompt += f"Label: \n"
        prompt += item_prompt
        answer = labels[item["Label"]]
        # sample = {"prompt": prompt, "answer": answer, "labels": labels}
        sample = {"prompt": prompt, "answer": answer}
        return sample
    def generate_labels(self,labels):
        choiceses=[]
        for label in labels:
            for subname in self.sub2label:
                if label in self.sub2label[subname].names:
                    # remove unlabeled
                    choiceses.append(self.sub2label[subname].names[1:])
                    break
        return choiceses
class TruthfulQADataset(Dataset):
    """TruthfulQA dataset from huggingface
    说明：
    """

    def __init__(self, ceval_path="", using_gpt=False, item_size=5):
        self.dataset = datasets.load_dataset("truthful_qa", "multiple_choice")[
            "validation"
        ]
        self.name = "TruthfulQA"
        self.item_size = item_size

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        ind = random.sample(range(len(self.dataset)), self.item_size)
        samples = self.dataset.select(ind)
        # Initialize the prompt string
        prompt = ""

        for i, sample in enumerate(samples):
            z = sample["mc1_targets"]
            
            # Shuffle choices and adjust answer accordingly
            shuffled_indices = list(range(len(z["choices"])))
            random.shuffle(shuffled_indices)
            shuffled_choices = [z["choices"][i] for i in shuffled_indices]
            new_answer_index = shuffled_indices.index(z["labels"].index(1))

            prompt += "Question: "+str(sample["question"]) + "\n"
            prompt += (
                "\n".join(
                    [f"{a}. {c}" for (a, c) in zip(ALPHABET[: len(shuffled_choices)], shuffled_choices)]
                )
                + "\n"
            )
            prompt += "Answer: " + ALPHABET[new_answer_index] + "\n"
            prompt += "\n"
        return prompt

    def __multi_choice_prompt__(self, ban_index=-1):
        ind = random.sample(range(len(self.dataset)), self.item_size)
        samples = self.dataset.select(ind)
        prompt = ""

        for i, sample in enumerate(samples):
            z = sample["mc2_targets"]
            prompt += "Question: "+str(sample["question"]) + "\n"
            prompt += (
                "\n".join(
                    [
                        f"{a}. {c}"
                        for (a, c) in zip(ALPHABET[: len(z["choices"])], z["choices"])
                    ]
                )
                + "\n"
            )
            ans = []
            for j, v in enumerate(sample["mc2_targets"]["labels"]):
                if v == 1:
                    ans.append(ALPHABET[int(j)])
            prompt += "Answer: " + ", ".join(ans) + "\n"
            prompt += "\n"
        return prompt

    def __getitem__(self, index):
        idx = index
        prompt = self.__generate_prompt__(idx)

        entry = self.dataset[idx]

        z = entry["mc1_targets"]
        # Shuffle choices and adjust answer accordingly
        shuffled_indices = list(range(len(z["choices"])))
        random.shuffle(shuffled_indices)
        shuffled_choices = [z["choices"][i] for i in shuffled_indices]
        new_answer_index = shuffled_indices.index(z["labels"].index(1))

        prompt += "Question: "+str(entry["question"]) + "\n"
        prompt += (
            "\n".join(
                [f"{a}. {c}" for (a, c) in zip(ALPHABET[: len(shuffled_choices)], shuffled_choices)]
            )
            + "\n"
        )
        prompt += "Answer: " +"\n"
        prompt += "\n"
        answer=ALPHABET[new_answer_index]
        sample = {"prompt": prompt, "answer": answer}
        return sample


class EPRSTMTDataset(Dataset):
    """EPRSTMT dataset from huggingface
    说明：
    """

    def __init__(self, ceval_path, using_gpt=False, item_size=5):
        dev_path = os.path.join(ceval_path,'dev_few_all.json')
        test_path = os.path.join(ceval_path,'test_public.json')
        # train集不读取
        with open(dev_path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.readlines()
            self.dev_dataset = list(map(lambda x: json.loads(x), data))



        with open(test_path, "r", encoding="utf-8") as f:
            data = f.readlines()
            self.dataset = list(map(lambda x: json.loads(x), data))
        self.name = "EPRSTMT"
        self.first_line = "判断以下段落情感是积极还是消极\n"
        self.item_size = item_size
        self.prompt_dict = {"Positive": "A", "Negative": "B"}

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        samples = random.sample(self.dev_dataset, self.item_size)
        # Initialize the prompt string
        prompt = self.first_line

        for i, sample in enumerate(samples):
            sentence = sample["sentence"]
            label = sample["label"]
            # 组合samples
            prompt += f"\n文本: {sentence}"
            prompt += f"\nA:积极"
            prompt += f"\nB:消极"
            prompt += f"\n答案:{self.prompt_dict[label]}\n"
            prompt += "\n"
        return prompt

    def __getitem__(self, index):
        idx = index
        prompt = self.__generate_prompt__(idx)
        # prompt = self.__multi_choice_prompt__(idx)
        question = self.dataset[idx]
        # z=entry['mc2_targets']
        sentence = question["sentence"]
        prompt += f"\n文本: {sentence}"
        prompt += f"\nA:积极"
        prompt += f"\nB:消极"
        prompt += f"\n答案:\n"

        sample = {"prompt": prompt, "answer": self.prompt_dict[question["label"]]}
        return sample


class TNEWSDataset(Dataset):
    """TNEWS dataset from huggingface
    说明：
    """

    def __init__(self, ceval_path, using_gpt=False, item_size=5):
        dev_path = os.path.join(ceval_path,'dev_few_all.json')
        test_path = os.path.join(ceval_path,'test_public.json')
        # train集不读取
        with open(dev_path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.readlines()
            self.dev_dataset = list(map(lambda x: json.loads(x), data))

        with open(test_path, "r", encoding="utf-8") as f:
            data = f.readlines()
            self.dataset = list(map(lambda x: json.loads(x), data))
        self.name = "TNEWS"
        self.first_line = "判断以下新闻属于哪一个类别\n"
        self.item_size = item_size
        self.prompt_dict = {
            # "news_house": "A",
            # "news_entertainment": "B",
            # "news_sports": "C",
            # "news_game": "D",
            # "news_military": "E",
            # "news_culture": "F",
            # "news_finance": "G",
            # "news_agriculture": "H",
            # "news_world": "I",
            # "news_travel": "J",
            # "news_tech": "K",
            # "news_story": "L",
            # "news_stock": "M",
            # "news_edu": "N",
            # "news_car": "O",
            "news_story": "A",
            "news_culture": "B",
            "news_entertainment": "C",
            "news_sports": "D",
            "news_finance": "E",
            "news_house": "F",
            "news_car": "G",
            "news_edu": "H",
            "news_tech": "I",
            "news_military": "J",
            "news_travel": "K",
            "news_world": "L",
            "news_stock": "M",
            "news_agriculture": "N",
            "news_game": "O",
        }

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        samples = random.sample(self.dev_dataset, self.item_size)
        # Initialize the prompt string
        prompt = self.first_line
        for i, sample in enumerate(samples):
            sentence = sample["sentence"]
            label = sample["label_desc"]
            keywords = sample["keywords"]
            # print(keywords)
            # 组合samples
            prompt += f"\n文本:"
            prompt += f"\n新闻标题: {sentence}"
            prompt += f"\n关键词: {keywords}"
            prompt += f"\nA:故事"
            prompt += f"\nB:文化"
            prompt += f"\nC:娱乐"
            prompt += f"\nD:体育"
            prompt += f"\nE:财经"
            prompt += f"\nF:房产"
            prompt += f"\nG:汽车"
            prompt += f"\nH:教育"
            prompt += f"\nI:科技"
            prompt += f"\nJ:军事"
            prompt += f"\nK:旅游"
            prompt += f"\nL:国际"
            prompt += f"\nM:股票"
            prompt += f"\nN:农业"
            prompt += f"\nO:电竞"
            prompt += f"\n答案:{self.prompt_dict[label]}\n"
            prompt += "\n"
        return prompt

    def __getitem__(self, index):
        idx = index
        prompt = self.__generate_prompt__(idx)
        # prompt = self.__multi_choice_prompt__(idx)
        question = self.dataset[idx]
        # z=entry['mc2_targets']
        sentence = question["sentence"]
        label = question["label_desc"]
        keywords = question["keywords"]
        prompt += f"\n文本:"
        prompt += f"\n新闻标题: {sentence}"
        prompt += f"\n关键词: {keywords}"
        prompt += f"\nA:故事"
        prompt += f"\nB:文化"
        prompt += f"\nC:娱乐"
        prompt += f"\nD:体育"
        prompt += f"\nE:财经"
        prompt += f"\nF:房产"
        prompt += f"\nG:汽车"
        prompt += f"\nH:教育"
        prompt += f"\nI:科技"
        prompt += f"\nJ:军事"
        prompt += f"\nK:旅游"
        prompt += f"\nL:国际"
        prompt += f"\nM:股票"
        prompt += f"\nN:农业"
        prompt += f"\nO:电竞"
        prompt += f"\n答案:\n"
        sample = {"prompt": prompt, "answer": self.prompt_dict[label]}
        return sample


class IMDBDataset(Dataset):
    """IMDB dataset from huggingface
    说明：
    """

    def __init__(self, ceval_path="", using_gpt=False, item_size=5):
        self.dataset = datasets.load_dataset("imdb")["test"]
        self.train_dataset = datasets.load_dataset("imdb")["train"]

        self.name = "IMDB"
        # self.first_line = "In this task, you will be presented with some text. Please determine whether the text is positive or negative.Please answer with 'Positive' or 'Negative'.\n"
        self.first_line = ""
        self.item_size = item_size
        self.prompt_dict = {1: "Positive", 0: "Negative"}

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        # ind = random.sample(range(len(self.dataset)), self.item_size)
        # samples = self.dataset.select(ind)
        # 换训练集
        ind = random.sample(range(len(self.train_dataset)), self.item_size)
        samples = self.train_dataset.select(ind)

        # Initialize the prompt string
        prompt = self.first_line

        for i, sample in enumerate(samples):
            prompt += "Passage: "+"\n"+str(sample["text"]) + "\n"
            # prompt += str(sample["text"]) + "\n"
            prompt += "Sentiment: " + self.prompt_dict[sample["label"]] + "\n"
            prompt += "\n"
        # print("prompt: ",prompt)
        return prompt

    def __getitem__(self, index):
        idx = index
        prompt = self.__generate_prompt__(idx)
        # prompt = self.__multi_choice_prompt__(idx)
        sample = self.dataset[idx]
        # z=entry['mc2_targets']
        answer = self.prompt_dict[sample["label"]]
        prompt += "Passage: "+"\n"+str(sample["text"]) + "\n"
        # prompt += str(sample["text"]) + "\n"
        prompt += "Sentiment: " + "\n"
        prompt += "\n"

        sample = {"prompt": prompt, "answer": answer}
        # print("sample: ",sample)
        return sample
    def generate_labels(self,labels):
        choiceses=[]
        for label in labels:
            choiceses.append(list(self.prompt_dict.values()))
        return choiceses

class BoolQDataset(Dataset):
    """BoolQ dataset from huggingface
    说明：
    """

    def __init__(self, ceval_path="", using_gpt=False, item_size=5):
        dataset = datasets.load_dataset("boolq")
        # print(dataset)
        self.name = "BoolQ"
        self.prompt_heads = [""]

        # 数据集文件是arrow文件，所以需要用datasets.load_from_disk，folder_path是数据集的文件夹路径
        self.item_size = item_size
        # self.prompt_dict = {1:'Positive', 0:'Negative'}
        self.choice = ["Yes", "No"]
        _content = []
        for k in range(len(dataset['validation'])):
            _content.append(dataset['validation'][k])
        self.dataset = _content
        _content = []
        for k in range(len(dataset['train'])):
            _content.append(dataset['train'][k])
        self.train_dataset = _content
    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        train_sample = random.sample(self.train_dataset, self.item_size)
        prompt = [random.choice(self.prompt_heads) + "\n"]
        for i, item in enumerate(train_sample):
            FLAG = str(item.get("answer", ""))
            if FLAG == "True":
                FLAG = "Yes"
            elif FLAG == "False":
                FLAG = "No"
            prompt_item = (
                "\nPassage: "
                + item["passage"]
                + "\nQuestion: "
                + item["question"]
                + "?"
                + "\nAnswer: "
                + FLAG
                + "\n"
            )
            prompt.append(prompt_item)
        prompt = "".join(prompt)
        return prompt

    def __getitem__(self, index):
        idx = index
        prompt = self.__generate_prompt__(idx)
        # prompt = self.__multi_choice_prompt__(idx)
        sample = self.dataset[idx]
        # z=entry['mc2_targets']
        # answer = self.prompt_dict[sample['answer']]
        FLAG = str(sample.get("answer", ""))
        if FLAG == "True":
            FLAG = "Yes"
        elif FLAG == "False":
            FLAG = "No"
        answer = FLAG
        prompt += (
            "\nPassage: "
            + sample["passage"]
            + "\nQuestion: "
            + sample["question"]
            + "?"
            + "\nAnswer: "
            + "\n"
        )

        sample = {"prompt": prompt, "answer": answer}
        return sample

class MMLUDataset(Dataset):
    """MMLU dataset from huggingface
    说明：施工完毕，可放心食用（
    """

    def __init__(self, ceval_path="", using_gpt=False, item_size=5):
        # dataset = load_dataset("tasksource/mmlu")
        dataset_name = "tasksource/mmlu"
        courses = [
            "abstract_algebra",
            "anatomy",
            "astronomy",
            "business_ethics",
            "clinical_knowledge",
            "college_biology",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_medicine",
            "college_physics",
            "computer_security",
            "conceptual_physics",
            "econometrics",
            "electrical_engineering",
            "elementary_mathematics",
            "formal_logic",
            "global_facts",
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            "high_school_european_history",
            "high_school_geography",
            "high_school_government_and_politics",
            "high_school_macroeconomics",
            "high_school_mathematics",
            "high_school_microeconomics",
            "high_school_physics",
            "high_school_psychology",
            "high_school_statistics",
            "high_school_us_history",
            "high_school_world_history",
            "human_aging",
            "human_sexuality",
            "international_law",
            "jurisprudence",
            "logical_fallacies",
            "machine_learning",
            "management",
            "marketing",
            "medical_genetics",
            "miscellaneous",
            "moral_disputes",
            "moral_scenarios",
            "nutrition",
            "philosophy",
            "prehistory",
            "professional_accounting",
            "professional_law",
            "professional_medicine",
            "professional_psychology",
            "public_relations",
            "security_studies",
            "sociology",
            "us_foreign_policy",
            "virology",
            "world_religions",
        ]
        self.name = "MMLU"
        self.prompt_heads=["The following are multiple choice questions (with answers) about "]# append with courses+'.'
        self.item_size = item_size
        self.choice = ["True", "False"]
        self.dataset,self.dev_dataset,self.sub2ind = self.load_datasets_parallel(courses=courses,dataset_name=dataset_name)

    def process_dataset(self,dataset_name, sub, val_content, dev_content, sub2ind):
        dataset = load_dataset(dataset_name, sub)
        for k in range(len(dataset["validation"])):
            self.val_content_lock.acquire()
            val_content.append(dataset["validation"][k])
            sub2ind.setdefault(sub,[]).append(self.i)
            self.i+=1
            self.val_content_lock.release()

        for k_dev in range(len(dataset["dev"])):
            dev_content.append(dataset["dev"][k_dev])

    def load_datasets_parallel(self,courses, dataset_name):
        self.sub2ind_lock = threading.Lock()
        self.val_content_lock = threading.Lock()

        self.i = 0
        val_content = []
        dev_content = []
        sub2ind = {}
        threads = []
        for sub in courses:
            thread = threading.Thread(target=self.process_dataset, args=(dataset_name, sub, val_content, dev_content, sub2ind))
            thread.start()
            threads.append(thread)

        # Wait for all threads to finish
        for thread in threads:
            thread.join()
        return val_content, dev_content, sub2ind
    
    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, val_question_index=-1):
        # pdb.set_trace()
        sub_name = ""
        for sub,indexs in self.sub2ind.items():
            # print("val_question_index: ",val_question_index)
            # print("indexs: ",indexs)
            # assert val_question_index in indexs
            if val_question_index in indexs:
                sub_name = sub
                break
        assert sub_name != ""
        train_sample = random.sample(self.dev_dataset,self.item_size)
        prompt = [random.choice(self.prompt_heads) + sub_name + ".\n\n"]
        # prompt = [random.choice(self.prompt_heads) + cource + ".\n\n"]
        for item in train_sample:
            choice = item["choices"]  # list of choices, number of choices varies
            # choice in prompt should have prefix of ABCE according to the number of choices
            prompt_choice = []
            for i in range(len(choice)):
                prompt_choice.append(f"{chr(65+i)}. {choice[i]}")
            prompt_choice = "\n".join(prompt_choice)
            Flag = ""

            Choice = []
            for i in range(len(choice)):
                Choice.append(f"{chr(65+i)}")
            if item.get("answer", "") != "":
                Flag = Choice[item.get("answer", "")]
            prompt_item = (
                f"Question: {item['question']}?\n{prompt_choice}\nAnswer: {Flag}\n\n"
            )
            prompt.append(prompt_item)

        prompt = "".join(prompt)
        return prompt

    def __getitem__(self, index):
        idx = index
        prompt = self.__generate_prompt__(idx)
        sample = self.dataset[idx]
        choice = sample["choices"]  # list of choices, number of choices varies
        # choice in prompt should have prefix of ABCE according to the number of choices
        prompt_choice = []
        for i in range(len(choice)):
            prompt_choice.append(f"{chr(65+i)}. {choice[i]}")
        prompt_choice = "\n".join(prompt_choice)
        Flag = ""

        Choice = []
        for i in range(len(choice)):
            Choice.append(f"{chr(65+i)}")
        if sample.get("answer", "") != "":
            Flag = Choice[sample.get("answer", "")]
        prompt_item = f"Question: {sample['question']}?\n{prompt_choice}\nAnswer: \n\n"
        prompt += prompt_item

        sample = {"prompt": prompt, "answer": Flag}
        return sample

class CMMLUDataset(Dataset):
    def __init__(self, ceval_path, using_gpt=False, item_size=5):
        with open(ceval_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        self.name = "CMMLU"
        self.first_line = "下面是一组问题，每个问题均有四个选项，请选出正确答案。\n"
        self.item_size = item_size

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        # Select random data samples from the dataset
        samples = random.sample(self.dataset, self.item_size)
        # Initialize the prompt string
        prompt = self.first_line
        for i, sample in enumerate(samples):
            # Add the sample information to the prompt
            prompt += "问题：" + str(sample["Question"]) + "\n"
            prompt += "A." + str(sample["choices"][0]) + "\n"
            prompt += "B." + str(sample["choices"][1]) + "\n"
            prompt += "C." + str(sample["choices"][2]) + "\n"
            prompt += "D." + str(sample["choices"][3]) + "\n"
            prompt += "答案：" + str(sample["Answer"]) + "\n"
            prompt += "\n"

        return prompt

    def __getitem__(self, index):
        sample = []
        idx = index
        prompt = self.__generate_prompt__(idx)
        entry = self.dataset[idx]
        answer = entry["Answer"]
        prompt += "问题：" + str(entry["Question"]) + "\n"
        prompt += "A." + str(entry["choices"][0]) + "\n"
        prompt += "B." + str(entry["choices"][1]) + "\n"
        prompt += "C." + str(entry["choices"][2]) + "\n"
        prompt += "D." + str(entry["choices"][3]) + "\n"
        prompt += "答案：" + "\n"
        prompt += "\n"

        sample = {"prompt": prompt, "answer": answer}
        return sample


class ChIDDataset(Dataset):
    """ChID dataset
    说明：
    """

    def __init__(self, ceval_path, using_gpt=False, item_size=5):
        
        dev_path = os.path.join(ceval_path,'dev_few_all.json')
        test_path = os.path.join(ceval_path,'test_public.json')
        # train集不读取
        with open(dev_path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.readlines()
            self.dev_dataset = list(map(lambda x: json.loads(x), data))
        
        self.name = "ChID"
        with open(test_path, "r", encoding="utf-8") as f:
            data = f.readlines()
            self.dataset = list(map(lambda x: json.loads(x), data))
        self.first_line = (
            # "在这个任务中，你将面对一些不完整的句子，其中句子中成语被'#idiom#'取代，以成语完形填空形式实现，从给定的七个选项,选择正确答案：\n"
            "阅读以下文章，并选择一个合适的成语\n"
        )
        self.item_size = item_size

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        # Select random data samples from the dataset
        samples = random.sample(self.dev_dataset, self.item_size)
        # Initialize the prompt string
        prompt = self.first_line

        for i, sample in enumerate(samples):
            # Add the sample information to the prompt
            prompt += "文本：" + str(sample["content"]) + "\n"
            prompt += "A." + str(sample["candidates"][0]) + "\n"
            prompt += "B." + str(sample["candidates"][1]) + "\n"
            prompt += "C." + str(sample["candidates"][2]) + "\n"
            prompt += "D." + str(sample["candidates"][3]) + "\n"
            prompt += "E." + str(sample["candidates"][4]) + "\n"
            prompt += "F." + str(sample["candidates"][5]) + "\n"
            prompt += "G." + str(sample["candidates"][6]) + "\n"

            prompt += "答案：" + chr(ord("A") + int(sample["answer"])) + "\n"
            prompt += "\n"

        return prompt

    def __getitem__(self, index):
        sample = []
        idx = index
        prompt = self.__generate_prompt__(idx)
        entry = self.dataset[idx]
        answer = chr(ord("A") + int(entry["answer"]))

        prompt += "文本：" + str(entry["content"]) + "\n"
        prompt += "A." + str(entry["candidates"][0]) + "\n"
        prompt += "B." + str(entry["candidates"][1]) + "\n"
        prompt += "C." + str(entry["candidates"][2]) + "\n"
        prompt += "D." + str(entry["candidates"][3]) + "\n"
        prompt += "E." + str(entry["candidates"][4]) + "\n"
        prompt += "F." + str(entry["candidates"][5]) + "\n"
        prompt += "G." + str(entry["candidates"][6]) + "\n"
        prompt += "答案：" + "\n"
        prompt += "\n"

        sample = {"prompt": prompt, "answer": answer}
        return sample


class CSLDataset(Dataset):
    """CSL dataset
    说明：
    """

    def __init__(self, ceval_path, using_gpt=False, item_size=5):
        self.name = "CSL"
        
        dev_path = os.path.join(ceval_path,'dev_few_all.json')
        test_path = os.path.join(ceval_path,'test_public.json')
        # train集不读取
        with open(dev_path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.readlines()
            self.dev_dataset = list(map(lambda x: json.loads(x), data))
        
        with open(test_path, "r", encoding="utf-8") as f:
            data = f.readlines()
            self.dataset = list(map(lambda x: json.loads(x), data))
        # self.first_line = "在此任务中，将给出一段摘要与几个关键词，根据给出的摘要与关键词的关系，判断关键词是真实还是伪造，关键词为真实时请回答'真实'，关键词为伪造时请回答'伪造'：\n"
        self.first_line = "摘要关键词判别\n\n"
        self.item_size = item_size
        self.prompt_dict = {"1": "正确", "0": "错误"}
        self.choice_dict = {"1": "A", "0": "B"}

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        # Select random data samples from the dataset
        samples = random.sample(self.dev_dataset, self.item_size)
        # Initialize the prompt string
        prompt = self.first_line
        choices=["A","B"]
        indexes=["0","1"]
        for i, sample in enumerate(samples):
            # Add the sample information to the prompt
            prompt += "文本: " + str(sample["abst"]) + "\n"
            prompt += "关键词：" + str(sample["keyword"]) + "\n"
            random.shuffle(indexes)
            prompt += choices[indexes.index("0")] + ". " + str(self.prompt_dict["0"]) + "\n"
            prompt += choices[indexes.index("1")] + ". " + str(self.prompt_dict["1"]) + "\n"
            # prompt += "答案：" + str(self.prompt_dict[str(sample["label"])]) + "\n"
            prompt += "答案: " + choices[indexes.index(str(sample["label"]))] + "\n"
            prompt += "\n"

        return prompt

    def __getitem__(self, index):
        sample = []
        idx = index
        prompt = self.__generate_prompt__(idx)
        entry = self.dataset[idx]
        choices=["A","B"]
        indexes=["0","1"]
        random.shuffle(indexes)
 
        # answer = str(self.prompt_dict[str(entry["label"])])
        answer = choices[indexes.index(str(entry["label"]))]

        prompt += "文本: " + str(entry["abst"]) + "\n"
        prompt += "关键词：" + str(entry["keyword"]) + "\n"
        prompt += choices[indexes.index("0")] + ". " + str(self.prompt_dict["0"]) + "\n"
        prompt += choices[indexes.index("1")] + ". " + str(self.prompt_dict["1"]) + "\n"
        prompt += "答案: " + "\n"
        prompt += "\n"

        sample = {"prompt": prompt, "answer": answer}
        return sample


class CLUEWSCDataset(Dataset):
    """CLUEWSC dataset
    说明：
    """

    def __init__(self, ceval_path, using_gpt=False, item_size=5):
        
        dev_path = os.path.join(ceval_path,'dev_few_all.json')
        test_path = os.path.join(ceval_path,'test_public.json')
        # train集不读取
        with open(dev_path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.readlines()
            self.dev_dataset = list(map(lambda x: json.loads(x), data))
        
        self.name = "CLUEWSC"
        with open(test_path, "r", encoding="utf-8") as f:
            data = f.readlines()
            self.dataset = list(map(lambda x: json.loads(x), data))
        self.first_line = "代词是否指向给定名词短语：\n"
        self.item_size = item_size
        self.prompt_dict = {"true": "A", "false": "B"}

    def __len__(self):
        return len(self.dataset)

    def __generate_prompt__(self, ban_index=-1):
        # Select random data samples from the dataset
        samples = random.sample(self.dev_dataset, self.item_size)
        # Initialize the prompt string
        prompt = self.first_line

        for i, sample in enumerate(samples):
            # Add the sample information to the prompt
            prompt += "段落：" + str(sample["text"]) + "\n"
            prompt += ("问题："
                       +str(sample["target"]["span1_text"])+'（'+str(sample["target"]["span1_index"])+'）'
                       +"是指代"
                       +str(sample["target"]["span2_text"])+'（'+str(sample["target"]["span2_index"])+'）'
                       +"吗\n"
                       +"A. 正确\n"
                       +"B. 错误\n")
            prompt += "答案：" + str(self.prompt_dict[str(sample["label"])]) + "\n"
            prompt += "\n"

        return prompt

    def __getitem__(self, index):
        sample = []
        idx = index
        prompt = self.__generate_prompt__(idx)
        entry = self.dataset[idx]
        answer = str(self.prompt_dict[str(entry["label"])])

        prompt += "" + str(entry["text"]) + "\n"
        prompt += ("问题："
                    +str(entry["target"]["span1_text"])+'（'+str(entry["target"]["span1_index"])+'）'
                    +"是指代"
                    +str(entry["target"]["span2_text"])+'（'+str(entry["target"]["span2_index"])+'）'
                    +"吗\n"
                    +"A. 正确\n"
                    +"B. 错误\n")
        prompt += "答案:\n"
        prompt += "\n"

        sample = {"prompt": prompt, "answer": answer}
        return sample
