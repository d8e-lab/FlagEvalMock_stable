import glob
import json
import os
import sys
from argparse import Namespace
import csv
import re

import torch
import transformers

from transformers.trainer_pt_utils import LabelSmoother

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "./")))

import argparse
# from model_zoo import MODEL_DICT

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
def postprocess(result):
    result = re.sub(r'<[^>]+>', ' ', result)
    result += '>'
    result_ = re.sub(r'<[^>]+>', ' ', result)
    if result_ != result and len(result_) > len(result) * 0.5:
        result = result_
    else:
        result = result[:-1]
    result = re.sub(r"^[\s\n\t:：\ufff0-\uffff]+", "", result)
    result = re.sub(r"[\s\n\t:：\ufff0-\uffff]+$", "", result)
    return result

def inference(model,tokenizer,context):
    prompt=context['prompt']
    inputs = tokenizer([prompt], padding=True, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.2,
            }
    output=model.generate(pad_token_id=tokenizer.pad_token_id,**inputs,**gen_kwargs)
    output=output[:, len(inputs["input_ids"][0]):]
    return tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)

def save_to_csv(filename,data):
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in data:
            csv_writer.writerow(row)

def main(args):
    # model_name = args.model_name
    # model_builder = MODEL_DICT[args.model_name]
    # llm = model_builder(args.model_name, args.model_path, args.tokenizer_path, gpu_id=0)

    print("loading model")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
    ).bfloat16().cuda().eval()
    print("loading tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        use_fast=False,
    )
    print("loading datasets")
    dataset_folder="/mnt/SFT_store/chinese_open_dataset"
    folders=os.listdir(dataset_folder)
    
    for folder in folders:
        dataset_path = os.path.join(dataset_folder, folder)
        for j_file in glob.iglob(os.path.join(dataset_path, "*.json")):
            print(j_file)
            with open(j_file, "r", encoding="utf-8") as f:
                data = f.readlines()
                dataset = list(map(lambda x: json.loads(x), data))
                i = 0
                data_item = []
                for context in dataset:
                    question=context['prompt']
                    context['prompt']=get_prompt(context['prompt'],[])
                    # print(context)
                    print(i)
                    output = inference(model,tokenizer,context)
                    output[0] = postprocess(output[0])
                    capability = os.path.split(j_file)[-1]
                    pattern = r'_0.(7|8).json'
                    capability = re.sub(pattern,'',capability)
                    i+=1
                    data_item.append([capability,question,context['answer'],output[0]])
                    if i==args.sample_num:
                        save_to_csv(args.output,data_item)
                        break
system_prompt = "你是思源，一个由厦门大学、北京大学深圳研究生院、合肥综合性国家科学中心人工智能研究院（安徽省人工智能实验室）、安徽淘云科技股份有限公司合作研发的人工智能助手。在保证安全的前提下，回答问题要尽可能有帮助。你的答案不应该包含任何有害的、不道德的、种族主义的、性别歧视的、有毒的、危险的或非法的内容。请确保你的回答在社会上是公正和积极的。如果一个问题没有任何意义，或者与事实不一致，解释为什么，而不是回答不正确的问题。如果你不知道问题的答案，请不要分享虚假信息。不要以xml的格式输出内容。"
# system_prompt_eng = "你是一个人工智能助手。在保证安全的前提下，回答问题要尽可能有帮助。你的答案不应该包含任何有害的、不道德的、种族主义的、性别歧视的、有毒的、危险的或非法的内容。请确保你的回答在社会上是公正和积极的。如果一个问题没有任何意义，或者与事实不一致，解释为什么，而不是回答不正确的问题。如果你不知道问题的答案，请不要分享虚假信息。"
# system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

# sen_test = DFA()

def get_prompt(message: str, chat_history: list[tuple[str, str]]) -> str:
    texts = []
    if len(chat_history) == 0:
        chat_history.append(('你是谁？', '我是思源，一个由厦门大学、北京大学深圳研究生院、合肥综合性国家科学中心人工智能研究院（安徽省人工智能实验室）、安徽淘云科技股份有限公司合作研发的人工智能助手。'))
        # if len(message) < 20:
        #     chat_history.append(('厦门大学的校长是谁？', '2022年起，厦门大学校长、党委副书记是张宗益'))
        #     chat_history.append(('厦门大学的党委书记是谁？', '2022年起，厦门大学党委书记是张荣'))
        #     chat_history.append(('介绍一下厦门大学的领导', '2022年起，厦门大学领导如下：党委书记张荣，校长、党委副书记张宗益，党委常务副书记林东伟，党委副书记全海，党委副书记徐进功。'))
    for i, (user_input, response) in enumerate(chat_history):
        if i == 0:
            texts.append(f'[INST]<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_input.strip()}[/INST]{response.strip()}')
        else:
            texts.append(f'[INST]{user_input.strip()}[/INST]{response.strip()}')
    if len(chat_history) == 0:
        texts.append(f'[INST]<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{message.strip()}[/INST]')
    else:
        texts.append(f'[INST]{message.strip()}[/INST]')
    return ''.join(texts)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",type=str)
    # parser.add_argument("--data_path",type=str)
    parser.add_argument("--output",type=str)
    parser.add_argument("--model_max_length",type=int)
    parser.add_argument("--sample_num",type=int)
    args = parser.parse_args()
    main(args)
