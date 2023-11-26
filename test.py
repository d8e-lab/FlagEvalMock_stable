from typing import List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, LlamaTokenizerFast
# from transformers import LlamaForCausalLM, LlamaTokenizerFast
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.generation.utils import GenerationConfig

import re
class BaiChuan2Chat:
    def __init__(self, model_name, model_path, tokenizer_path, config_path="",gpu_id=0) -> None:
        self.name = model_name
        # self.generation_config=GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-7B-Chat")
        # print(type(self.generation_config),self.generation_config)
        # exit()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).bfloat16().to(gpu_id).eval()
        # self.model.generation_config = self.generation_config
        self.device=self.model.device

    def inference(self, queries, choiceses, use_logits, nt, dataset_name):
        max_length=2048
        generation_config = GenerationConfig(
            **{
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "user_token_id": 195,
                "assistant_token_id": 196,
                "max_new_tokens": nt,
                "temperature": 0.3,
                "top_k": 5,
                "top_p": 0.85,
                "repetition_penalty": 1.05,
                "do_sample": False,
                "transformers_version": "4.29.2"
            }
        )
        responses=[]
        for query in queries:
            query=preprocess(query)
            # messages = []
            # messages.append({"role": "user", "content": query})
            # response = self.model.chat(self.tokenizer, messages)

            # input_ids = self.build_chat_input(self, self.tokenizer, messages, generation_config.max_new_tokens)
            
            input_ids=[generation_config.user_token_id]
            input_ids+= self.tokenizer.encode(query)
            input_ids+=[generation_config.assistant_token_id]
            if(len(input_ids)>=max_length): 
                input_ids=[generation_config.user_token_id]+input_ids[-max_length:]
            input_ids = torch.LongTensor([input_ids]).to(self.device)
            outputs = self.model.generate(input_ids, generation_config=generation_config)
            response = self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

            responses.append(postprocess(response)) # 'Positive', 'Negative']
        return responses
    # def inference(self, queries, choiceses, use_logits, nt, dataset_name):
    #     choice_tokenss = []
    #     for choices in choiceses:
    #         choice_tokens = [
    #             self.tokenizer.encode(choice, add_special_tokens=False)[0]
    #             for choice in choices
    #         ]
    #         choice_tokenss.append(choice_tokens)
    #     queries = [preprocess(q) for q in queries]
    #     inputs = self.tokenizer(queries,return_tensors='pt',max_length=2048).to(self.model.device)
    #     gen_kwargs = {
    #         "max_new_tokens": nt,
    #         "repetition_penalty":1.1,
    #     }
    #     outputs = self.model.generate(**inputs,**gen_kwargs)
    #     outputs = outputs[:,-1*nt:]
    #     results = self.tokenizer.batch_decode(
    #         outputs,
    #         skip_special_tokens=True
    #     )
    #     return results