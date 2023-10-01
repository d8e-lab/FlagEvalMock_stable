from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, LlamaTokenizerFast
# from transformers import LlamaForCausalLM, LlamaTokenizerFast
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
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
        # print('model_name:',model_name)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True) #.half().cuda() #
            .bfloat16()
            .to(gpu_id)
        )
        if "chatglm2" not in self.name:
            self.tokenizer.pad_token = self.tokenizer.bos_token
            self.model.config.pad_token_id = self.model.config.bos_token_id
        pass

    def inference(self, queries, choiceses, use_logits, nt, dataset_name):
        choice_tokenss = []
        for choices in choiceses:
            choice_tokens = [
                self.tokenizer.encode(choice, add_special_tokens=False)[0]
                for choice in choices
            ]
            choice_tokenss.append(choice_tokens)
        queries = [preprocess(q) for q in queries]
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
            } 
            # 去除token_type_ids
            # inputs = {key:value for key,value in inputs.items() if key!='token_type_ids'}
            outputs = self.model.generate(pad_token_id=self.tokenizer.pad_token_id,**inputs, **gen_kwargs)
            
            outputs = outputs[:,-1*nt:]
            

            results = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            assert len(results) == len(queries)
            
        results = [postprocess(result,self.name) for result in results]
    
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path ,padding_side='left',truncation_side="left" , trust_remote_code=True
        )
        peft_config = PeftConfig.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, 
                                                          trust_remote_code=True)
        self.model =  PeftModel.from_pretrained(self.model,model_path,).bfloat16().to(gpu_id)
        if "chatglm2" not in self.name:
            self.tokenizer.pad_token = self.tokenizer.bos_token
            self.model.config.pad_token_id = self.model.config.bos_token_id

# if __name__ == "__main__":
    # model = Llama2_Lora(model_name="llama2_lora",model_path="/mnt/SFT_store/xxw/outputs/5peft/item0",tokenizer_path="/mnt/SFT_store/xxw/outputs/5peft/item0")
 
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
}
