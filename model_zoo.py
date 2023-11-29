from typing import List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, LlamaTokenizerFast
# from transformers import LlamaForCausalLM, LlamaTokenizerFast
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.generation.utils import GenerationConfig

import re

# def preprocess(query):
#     # query=query.strip("\n ：:")
#     # query+=":"
#     query=query.strip("\s\n")
#     return query
def preprocess(query):
    query=re.sub(r"^[\s\n\t\ufff0-\uffff]+", "", query)
    query=re.sub(r"[\s\n\t\ufff0-\uffff]+$", "", query)
    return query

# def postprocess(result,model_name:str=""):
#     result = result.split("\n")[0]
#     # result = result.split(' ', 1)[0]
#     result=result.strip("答案")
#     result=result.replace("Sentence","")
#     result=result.replace("Passage","")
#     result=result.replace("Question","")
#     result = re.sub(r"^[\s\n\t:：\ufff0-\uffff]+", "", result)
#     result = re.sub(r"[\s\n\t:：\ufff0-\uffff]+$", "", result)
#     return result
def postprocess(result,model_name:str=""):
    result = result.split("\n")[0]
    result=result.strip("答案")
    result = re.sub(r"^[\s\n\t:：\ufff0-\uffff]+", "", result)
    result = re.sub(r"[\s\n\t:：\ufff0-\uffff]+$", "", result)
    result = re.sub(r'</s>', '', result)
    result = re.sub(r'^[ ]+', '', result)
    result = re.sub(r'iment: ','',result) if model_name=='InternLM' else result
    return result

def postprocess_(result,model_name:str=""):
    # result = result.split("\n")[0]
    result=result.strip("答案")
    result=result.replace("Sentence","")
    result=result.split("Paper",1)[0]
    result=result.split("Tweet",1)[0]
    result=result.split("Article",1)[0]
    result=result.split("Query",1)[0]
    result=result.split("Tile",1)[0]
    result=result.split("Text",1)[0]
    result=result.split(":",1)[0]
    result=result.replace("Passage","")
    result=result.replace("Question","")
    result = re.sub(r"^[\s\n\t:：\ufff0-\uffff]+", "", result)
    result = re.sub(r"[\s\n\t:：\ufff0-\uffff]+$", "", result)
    return result

class BaseLLM:
    def __init__(self) -> None:
        self.name = "BaseLLM"
        self.tokenizer = None
        self.model = None
    def get_token_length(self, labels):
        if isinstance(labels, list):
            return max([len(self.tokenizer(l).input_ids) for l in labels])
        elif isinstance(labels, dict):
            return max([len(self.tokenizer(l).input_ids) for l in labels.values()])

class Llama2(BaseLLM):
    def __init__(self, model_name, model_path, tokenizer_path, config_path="",gpu_id=0) -> None:
        self.name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path ,padding_side='left',truncation_side="left" , trust_remote_code=True
        )
        print('gpu_id:',gpu_id)            
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True) #.half().cuda() #
            .bfloat16()
            .to(gpu_id).eval()
        )
        if "chatglm2" not in self.name:
            self.tokenizer.pad_token = self.tokenizer.bos_token
            self.model.config.pad_token_id = self.model.config.bos_token_id
        pass

    def inference(self, queries, choiceses, use_logits, nt, dataset_name=""):
        choice_tokenss = []
        for choices in choiceses:
            choice_tokens = [
                self.tokenizer.encode(choice, add_special_tokens=False)[0]
                for choice in choices
            ]
            choice_tokenss.append(choice_tokens)
        queries = [preprocess(q) for q in queries]
        inputs = self.tokenizer(queries, padding=True, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
        # print(inputs)
        if use_logits:
            results = []
            outputs = self.model(inputs.input_ids)# return_last_logit=True)
            logits = outputs.logits[:, -1]
            for i, (choice_tokens,choices) in enumerate(zip(choice_tokenss,choiceses)):
                logit = logits[i, choice_tokens] # 选取对应choice index的logit
                p = logit.argmax(dim=-1)
                results.append(choices[p])
        else:
            gen_kwargs = {
                "max_new_tokens": nt,
                "num_beams": 1,
                "do_sample": False,
                "top_p": 0.9,
                "temperature": 0.1,
            } 
            # 去除token_type_ids
            inputs = {key:value for key,value in inputs.items() if key!='token_type_ids'}
            outputs = self.model.generate(pad_token_id=self.tokenizer.pad_token_id,**inputs, **gen_kwargs)
            # print(len(inputs["input_ids"][0]),len(inputs["input_ids"][0]))
            # outputs = outputs[:,-1*nt:]
            outputs = outputs[:,len(inputs["input_ids"][0]):]

            results = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            # assert len(results) == len(queries)
            
        # results = [postprocess(result,self.name) for result in results]
    
        return results

    def chat(self, queries, choiceses, use_logits, nt, dataset_name=""):
        # choice_tokenss = []
        # for choices in choiceses:
        #     choice_tokens = [
        #         self.tokenizer.encode(choice, add_special_tokens=False)[0]
        #         for choice in choices
        #     ]
        #     choice_tokenss.append(choice_tokens)
        # queries = [preprocess(q) for q in queries]
        inputs = self.tokenizer(queries, padding=True, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
        # if use_logits:
        #     results = []
        #     outputs = self.model(inputs.input_ids)# return_last_logit=True)
        #     logits = outputs.logits[:, -1]
        #     for i, (choice_tokens,choices) in enumerate(zip(choice_tokenss,choiceses)):
        #         logit = logits[i, choice_tokens] # 选取对应choice index的logit
        #         p = logit.argmax(dim=-1)
        #         results.append(choices[p])
        # else:
        gen_kwargs = {
            "max_new_tokens": nt,
            "num_beams": 1,
            "do_sample": False,
            "top_p": 0.9,
            "temperature": 0.1,
        } 
        # 去除token_type_ids
        # inputs = {key:value for key,value in inputs.items() if key!='token_type_ids'}
        print(inputs["input_ids"].shape)
        print(inputs["input_ids"])
        outputs = self.model.generate(pad_token_id=self.tokenizer.pad_token_id,**inputs, **gen_kwargs)
        print(outputs)
        print(outputs.shape)
        # outputs = outputs[:,-1*nt:]   
        outputs = outputs[:,inputs["input_ids"].shape[1]:]
        

        results = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        assert len(results) == len(queries)
            
        # results = [postprocess(result,self.name) for result in results]
    
        return results

MODEL_CONFIGS = {
    '7b': LlamaConfig(num_hidden_layers=32+4, vocab_size=49953),
    '13b': LlamaConfig(hidden_size=5120, intermediate_size=13760, num_hidden_layers=40, num_attention_heads=40),
    '30b': LlamaConfig(hidden_size=6656, intermediate_size=17888, num_hidden_layers=60, num_attention_heads=52),
    '65b': LlamaConfig(hidden_size=8192, intermediate_size=22016, num_hidden_layers=80, num_attention_heads=64),
}


class Llama_colossalai:
    def __init__(self, model_name, model_path, tokenizer_path, config_path="",gpu_id=0) -> None:
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True)
            .bfloat16()
            .to(gpu_id)
        )
          
        self.name = model_name
        self.tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_path, padding_side='left',truncation_side="left", trust_remote_code=True
        )
        if "chatglm2" not in self.name:
            self.tokenizer.pad_token = self.tokenizer.bos_token
            self.model.config.pad_token_id = self.model.config.bos_token_id
        pass
    def inference(self, queries, choiceses, use_logits, nt,dataset_name):
        choice_tokenss = []
        for choices in choiceses:
            choice_tokens = [
                self.tokenizer.encode(choice, add_special_tokens=False)[0]
                for choice in choices
            ]
            choice_tokenss.append(choice_tokens)
        queries = [preprocess(q) for q in queries]
        for i in range(len(queries)):
            queries[i] = queries[i].replace('\n', ' ')
            queries[i] = queries[i].replace('\\', '')
        inputs = self.tokenizer(queries, padding=True, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
        if use_logits:
            results = []
            outputs = self.model(inputs.input_ids)# return_last_logit=True)
            logits = outputs.logits[:, -1]
            for i, (choice_tokens,choices) in enumerate(zip(choice_tokenss,choiceses)):
                logit = logits[i, choice_tokens] # 选取对应choice index的logit
                p = logit.argmax(dim=-1)
                results.append(choices[p])
        else:
            gen_kwargs = {
                "max_new_tokens": nt,
                "num_beams": 1,
                "do_sample": False,
                "top_p": 0.9,
                "temperature": 0.1,
            }  # "logits_processor": logits_processor, **kwargs}
            outputs = self.model.generate(pad_token_id=self.tokenizer.pad_token_id,**inputs, **gen_kwargs)
            if dataset_name in ['MMLU','IMDB','RAFT','C-Eval']:
                outputs = outputs[:,-1*nt:]
            else:
                outputs = outputs[:,-1*nt]
            results = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        print("outputs: "+results[0])
        if dataset_name in ['MMLU','RAFT','C-Eval']:
            results = [postprocess_(result,self.name) for result in results]
        else:
            results = [postprocess(result,self.name) for result in results]
        return results
    
class Llama2_Lora(Llama2):
    def __init__(self, model_name, model_path, tokenizer_path, config_path="",gpu_id=0) -> None:
        # 为兼容用法 此处model_path实则为peft_path
        from peft import PeftConfig,PeftModel
        self.name = model_name
        peft_config = PeftConfig.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, 
                                                          trust_remote_code=True)
        self.model =  PeftModel.from_pretrained(self.model,model_path,).bfloat16().to(gpu_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            peft_config.base_model_name_or_path ,padding_side='left',truncation_side="left" , trust_remote_code=True
        )
        if "chatglm2" not in self.name:
            self.tokenizer.pad_token = self.tokenizer.bos_token
            self.model.config.pad_token_id = self.model.config.bos_token_id


class Llama2_GLora(Llama2):
    def __init__(self, model_name, base_path,  tokenizer_path, config_path="",gpu_id=0) -> None:
        from peft import PeftConfig,PeftModel
        from peft_utils import set_glora,load_glora,set_glora_eval_config
        import torch
        self.name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_path ,padding_side='left',truncation_side="left" , trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(base_path, 
                                                          trust_remote_code=True)
        load_path="/mnt/SFT_store/flageval_peft/outputs/glora/2023-10-16_18-03-57_success/final.pt"
        set_glora(self.model,4)
        load_glora(load_path,self.model)
        self.model = self.model.bfloat16().to(gpu_id).eval()
        config_path ="/mnt/SFT_store/flageval_peft/outputs/glora/evolution_2/checkpoint-20.pth.tar"
        # config_path ="/mnt/SFT_store/flageval_peft/outputs/glora/search/2023-10-27_04-58-43/checkpoint-20.pth.tar"
        info = torch.load(config_path)
        eval_config =info['keep_top_k'][50][int(tokenizer_path)]
        set_glora_eval_config(eval_config=eval_config,model=self.model)
        if "chatglm2" not in self.name:
            self.tokenizer.pad_token = self.tokenizer.bos_token
            self.model.config.pad_token_id = self.model.config.bos_token_id
            
# if __name__=="__main__":
#     a=Llama2_GLora("name","/mnt/SFT_store/Linksoul-llama2-7b","/mnt/SFT_store/Linksoul-llama2-7b")

class Llama2_repadapter(Llama2):
    def __init__(self, model_name, base_path, repadapter_path, config_path="",gpu_id=0) -> None:
        from peft_utils import set_repadapter,load_repadapter
        self.name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_path ,padding_side='left',truncation_side="left" , trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(base_path, 
                                                          trust_remote_code=True).bfloat16()
        load_path = repadapter_path
        set_repadapter(self.model)
        load_repadapter(load_path,self.model)
        self.model = self.model.bfloat16().to(gpu_id)
        if "chatglm2" not in self.name:
            self.tokenizer.pad_token = self.tokenizer.bos_token
            self.model.config.pad_token_id = self.model.config.bos_token_id 

class Llama2_ssf(Llama2):
    def __init__(self, model_name, base_path, load_path, config_path="",gpu_id=0) -> None:
        from peft_utils import set_ssf,load_ssf
        self.name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_path ,padding_side='left',truncation_side="left" , trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(base_path, 
                                                          trust_remote_code=True).bfloat16()
        load_path = load_path
        set_ssf(self.model)
        load_ssf(load_path,self.model)
        self.model = self.model.bfloat16().to(gpu_id)
        if "chatglm2" not in self.name:
            self.tokenizer.pad_token = self.tokenizer.bos_token
            self.model.config.pad_token_id = self.model.config.bos_token_id 


class Qwen(Llama2):
    def __init__(self, model_name="", base_path="", peft_path="", tokenizer_path="", config_path="",gpu_id=0) -> None:
        from transformers.generation import GenerationConfig
        self.name = model_name
        # cache_path = "/home/xxw/.cache/huggingface/hub/models--Qwen--Qwen-7B/snapshots/c6bf4b5d52d7f81dbd6f046eb7efacc2ce3dae2b"
        cache_path = "/data/LLM/c6bf4b5d52d7f81dbd6f046eb7efacc2ce3dae2b/"
        cache_path = "/mnt/SFT_store/LLM/Qwen-7B/"
        self.tokenizer = AutoTokenizer.from_pretrained(cache_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(cache_path, trust_remote_code=True).bfloat16().to(gpu_id).eval()
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id = 151643 # [] change to bos
        self.model.generation_config = GenerationConfig.from_pretrained(cache_path, trust_remote_code=True)
    def inference(self, queries, choiceses, use_logits, nt, dataset_name):
        responses=[]
        for query in queries:
            query=preprocess(query)
            inputs = self.tokenizer(query,padding=False, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
            # print(inputs)
            response = self.model.generate(**inputs)
            response = response.cpu()[0][:, len(inputs["input_ids"]):]
            response=self.tokenizer.decode(response, skip_special_tokens=True)
            responses.append(postprocess(response))
        return responses

class BaiChuan2Base(Llama2):
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
                input_ids=[generation_config.user_token_id]+input_ids[-max_length+1:]
            input_ids = torch.LongTensor([input_ids]).to(self.device)
            outputs = self.model.generate(input_ids, generation_config=generation_config)
            response = self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

            responses.append(postprocess(response))
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

class BaiChuan2Chat(Llama2):
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


class InternLM(BaseLLM):
    def __init__(self, model_name, model_path, tokenizer_path, config_path="", gpu_id=0) -> None:
        self.gpu_id=gpu_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True,padding_side='left',truncation_side="left" , trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True, trust_remote_code=True).bfloat16().cuda(gpu_id).eval()    
    
    @torch.no_grad()
    def inference(self, queries,choiceses=None, use_logits=False, nt=50, dataset_name=None):
        results = []
        gen_kwargs = {
            # "max_length": 128,
            "max_new_tokens": nt, 
            "top_p": 0.9, 
            "num_beams": 1,
            "temperature": 0.1, 
            "do_sample": False, 
            "repetition_penalty": 1.05
        }
        queries = [preprocess(q) for q in queries]
        
        # inputs = self.tokenizer(queries,padding=True,truncation=True, return_tensors="pt",max_length=8192).to(self.model.device)
        # outputs = self.model.generate(**inputs, **gen_kwargs)
        # outputs = outputs[:,-1*nt:]
        # results = self.tokenizer.batch_decode(outputs,skip_special_tokens=True,
        #                                       clean_up_tokenization_spaces=False)
        for query in queries:
            # print("query: ",query)
            inputs = self.tokenizer([query], return_tensors="pt", truncation=False, padding=True, max_length=8192)
            if len(inputs) > 8192:
                # cut_len=8192-len(self.start_token["input_ids"][0])
                inputs = {k: v[:,-1*8192:] for k, v in inputs.items() if torch.is_tensor(v)}
                # inputs = {k: torch.cat((self.start_token[k],v),dim=1) for k, v in inputs.items() if torch.is_tensor(v)}
            inputs = {k: v.to(self.model.device) for k, v in inputs.items() if torch.is_tensor(v)}
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[0].cpu().tolist()
            outputs = outputs[len(inputs["input_ids"][0]):]
            result = self.tokenizer.decode(outputs, skip_special_tokens=True)
            results.append(result)
        results = [postprocess(result,self.__class__.__name__) for result in results]
        return results

class InternLM_Chat(BaseLLM):
    def __init__(self, model_name, model_path, tokenizer_path, config_path="", gpu_id=0) -> None:
        super().__init__()
        self.name = "InternLM"
        # self.history = []
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True,padding_side='left',truncation_side="left" , trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True, trust_remote_code=True).bfloat16().cuda(gpu_id).eval()
        
        # Setting padding tokens if required
        self.tokenizer.pad_token = self.tokenizer.bos_token
        self.model.config.pad_token_id = self.model.config.bos_token_id
        self.device=self.model.device
        self.start_token=self.tokenizer(["""<|User|>:"""], return_tensors="pt")

    def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = [],max_length:int=4096):
        prompt = ""
        for record in history:
            prompt += f"""<|User|>:{record[0]}<eoh>\n<|Bot|>:{record[1]}<eoa>\n"""
        prompt += f"""<|User|>:{query}<eoh>\n<|Bot|>:"""
        return tokenizer([prompt], return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    
    @torch.no_grad()
    def inference(self, queries,choiceses=None, use_logits=False, nt=50, dataset_name=None):
        results = []
        gen_kwargs = {
            "max_new_tokens": nt,
            "num_beams": 1,
            "do_sample": False,
            "top_p": 0.9,
            "temperature": 0.1,
        }    
        before_post=[]
        queries = [preprocess(q) for q in queries]
        for query in queries:
            prompt = f"""<|User|>:{query}<eoh>\n<|Bot|>:"""
            inputs =  self.tokenizer([prompt], return_tensors="pt", truncation=True, padding=True, max_length=2048)
            # if inputs["input_ids"].shape[1] > 2048:
            if len(inputs["input_ids"][0]) > 2048:
                cut_len=2048-len(self.start_token["input_ids"][0])
                inputs = {k: v[:,-1*cut_len:] for k, v in inputs.items() if torch.is_tensor(v)}
                inputs = {k: torch.cat((self.start_token[k],v),dim=1) for k, v in inputs.items() if torch.is_tensor(v)}
            inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
            outputs = self.model.generate(**inputs, **gen_kwargs)
            
            outputs = outputs[0].cpu().tolist()[len(inputs["input_ids"][0]):]
            response = self.tokenizer.decode(outputs, skip_special_tokens=True)
            before_post.append(response)
            response = response.split("<eoa>")[0]
            # history = history + [(query, response)]
            # return response, history
            results.append(response)
        # print(results)
        results = [postprocess(result,self.name) for result in results]
        # return results,before_post
        return results

class AquilaChat(BaseLLM):
    def __init__(self, model_name, model_path, tokenizer_path, config_path="", gpu_id=0) -> None:
        super().__init__()
        self.name=model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True) #.bfloat16().to(gpu_id).eval()
        self.model = self.model.eval().half().to(gpu_id)
        self.device = self.model.device
    @torch.no_grad()
    def inference(self, queries, choiceses, use_logits=False, nt=50, dataset_name=None):
        max_length=2048
        results = []
        generation_config= GenerationConfig(
            **{
                "_from_model_config": True,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 0,
                # "transformers_version": "4.28.1",
                "max_new_tokens": nt,
            }
        )
        for query in queries:
            # print(type(query),query)
            tokens = self.tokenizer.encode_plus(text=query,
                                                max_length=max_length,
                                                truncation="longest_first",
                                                )['input_ids'][:-1]
            tokens = torch.tensor(tokens)[None,].to(self.device)

            stop_tokens = ["###", "[UNK]", "</s>"]
            out = self.model.generate(tokens, do_sample=False, eos_token_id=100007, 
                                 bad_words_ids=[[self.tokenizer.encode(token)[0] for token in stop_tokens]],
                                 generation_config=generation_config)[0]
            # print("TOKEN length ",len(out)," - ",tokens.shape)
            out=out[tokens.shape[1]:]
            out = self.tokenizer.decode(out.cpu().numpy().tolist())
            results.append(out)
        results = [postprocess(result,self.name) for result in results]
        return results
    
class YulanChat(Llama2):
    def __init__(self, model_name, model_path, tokenizer_path, config_path="", gpu_id=0) -> None:
        self.name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained("/mnt/SFT_store/LLM/YuLan-Chat-2-13b-fp16")
        self.model = (
           AutoModelForCausalLM.from_pretrained("/mnt/SFT_store/LLM/YuLan-Chat-2-13b-fp16").half().to(gpu_id).eval()
        )    
        
    def inference(self, queries, choiceses, use_logits, nt, dataset_name=""):
        choice_tokenss = []
        for choices in choiceses:
            choice_tokens = [
                self.tokenizer.encode(choice, add_special_tokens=False)[0]
                for choice in choices
            ]
            choice_tokenss.append(choice_tokens)
        queries = [preprocess(q) for q in queries]
        # queries = [ f"""[|Human|]:{q}\n[|AI|]:""" for q in queries]
        # print(queries)
        inputs = self.tokenizer(queries, padding=True, return_tensors="pt", truncation=True, max_length=8192,return_attention_mask=True).to(self.model.device)
        # for input in inputs:
            # print(self.tokenizer.decode(input))
        if use_logits:
            results = []
            outputs = self.model(inputs.input_ids)# return_last_logit=True)
            logits = outputs.logits[:, -1]
            for i, (choice_tokens,choices) in enumerate(zip(choice_tokenss,choiceses)):
                logit = logits[i, choice_tokens] # 选取对应choice index的logit
                p = logit.argmax(dim=-1)
                results.append(choices[p])
        else:
            gen_kwargs = {
                "_from_model_config": True,
                "max_new_tokens": nt,
                "num_beams": 1,
                "do_sample": False,
                "top_p": 0.9,
                "temperature": 0.1,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 0,
                "transformers_version": "4.28.1"
            } 
            # 去除token_type_ids
            # inputs = {key:value for key,value in inputs.items() if key!='token_type_ids'}
            outputs = self.model.generate(**inputs, **gen_kwargs)
            
            outputs = outputs[:,-1*nt:]
            

            results = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            assert len(results) == len(queries)
            
        results = [postprocess(result,self.name) for result in results]
    
        return results
    # def inference(self, queries, choiceses, use_logits, nt, dataset_name=""):
    #     def postprocess(result, name):
    #         # 将result转换为字符串，去除首尾的空格
    #         result = str(result).strip()
            
    #         # 切分文本以"[|AI|]:"为标记
    #         split_result = result.split("[|AI|]:")
            
    #         if len(split_result) > 1:
    #             # 获取并去除首尾空格
    #             result = split_result[1].strip()
    #         else:
    #             result = ""  # 如果未找到，则返回空字符串
    #         result = re.sub(r"^[\s\n\t:：\ufff0-\uffff]+", "", result)
    #         result = re.sub(r"[\s\n\t:：\ufff0-\uffff]+$", "", result)
    #         result = re.sub(r'</s>', '', result)
    #         return result
    #     choice_tokenss = []
    #     for choices in choiceses:
    #         choice_tokens = [
    #             self.tokenizer.encode(choice, add_special_tokens=False)[0]
    #             for choice in choices
    #         ]
    #         choice_tokenss.append(choice_tokens)
    #     queries = [preprocess(q) for q in queries]
    #     queries = f"""[|Human|]:{queries}\n[|AI|]:<\s>"""
    #     inputs = self.tokenizer(queries, padding=True, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
    #     gen_kwargs = {
    #         "max_new_tokens":nt,
    #         "do_sample": False,
    #         "bos_token_id": 1,
    #         "eos_token_id": 2,
    #         "pad_token_id": 0,
            
    #     } 
    #     # 去除token_type_ids
    #     # inputs = {key:value for key,value in inputs.items() if key!='token_type_ids'}
    #     outputs = self.model.generate(**inputs, **gen_kwargs)
        
    #     outputs = outputs[:,-1*nt:]
        

    #     results = self.tokenizer.batch_decode(
    #         outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
    #     )
    #     # print(type(results)) list ['',]
    #     # print(type(queries)) str
    #     # print(results[0])
    #     # print(results[1])
    #     # print(queries)
    #     print(results)
    #     # assert len(results) == len(queries)
            
    #     results = [postprocess(result,self.name) for result in results]
    #     print(results)
    #     return results

MODEL_DICT = {
    # "chatglm2-6b": ChatGLM2,
    "chatglm2-6b": Llama2,
    # "linlyllama2-7b": LinlyLLaMA2,
    "llama": Llama2,
    "llama2":Llama2,
    "llama2chat":Llama2,
    "llama_colossalai":Llama_colossalai,
    "LinkSoul_base":Llama2,
    "LinkSoul_chat":Llama2,
    "LinkSoul_base_left":Llama2,
    "LinkSoul_chat_left":Llama2,
    "LinkSoul_1_1_base":Llama2,
    "LinkSoul_1_1_chat":Llama2,
    "LinkSoul_sft":Llama2,
    "Michael_base":Llama2,
    "Michael_v03":Llama2,
    "mac_llm":Llama_colossalai,
    "llama2_lora":Llama2_Lora,
    "llama2_glora":Llama2_GLora,
    "Qwen":Qwen,
    "BaiChuan2_base":BaiChuan2Base,
    "BaiChuan2_chat":BaiChuan2Chat,
    "InternLM":InternLM,
    "AquilaChat":AquilaChat,
    "YulanChat":YulanChat,
    "llama2_repadapter":Llama2_repadapter,
    "llama2_ssf":Llama2_ssf,
}
