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
def postprocess(result,model_name:str=""):
    result = result.split("\n")[0]
    result=result.strip("答案")
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
        # print('>>>queries:\n',queries[0],'\nqueries end<<<')
        # test_outputs=[self.tokenizer.encode(q, add_special_tokens=False) for q in queries]
        # for o in test_outputs:
        #     print("ENCODE LEN: " ,type(o), len(o))
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
            # inputs.input_ids
            # if self.name=='LinkSoul_base':
            # 去除token_type_ids
            inputs = {key:value for key,value in inputs.items() if key!='token_type_ids'}
                # print('>>>inputs(model_zoo.py) after process: ',inputs,'<<<')
            outputs = self.model.generate(pad_token_id=self.tokenizer.pad_token_id,**inputs, **gen_kwargs)
            # outputs = self.model.generate(**inputs, **gen_kwargs)
            # 【】 In case query that are over length than 2048, we directly use the last nt tokens as the result
            # print("OUTPUTS: ", outputs.shape)
            # for ii in inputs.input_ids:
            #     print("INPUTS: ", ii.shape)
            outputs = outputs[:,-1*nt:]
            # print("OUTPUTS: ", outputs.shape)

            results = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            assert len(results) == len(queries)
            # for o,r in zip(outputs,results):
            #     print("OR: ", o," | ", r)
            #     # convert r to hex
            #     # r = r.encode("utf-8")
            #     # r = r.decode("utf-8")
            # exit()          
            # # 【】 For query that are over length than 2048, the result will be wrong
            # for i,(query, result) in enumerate(zip(queries, results)):
            #     print(f"{i}: {len(query)} | {len(result)}")
            #     length = len(query)
            #     results[i] = result[length:]
            # result = []
            # for idx in range(len(outputs)):
            #     # output = outputs.tolist()[idx][-1*nt:]
            #     output = outputs.tolist()[idx][-1*nt:]
            #     response = self.tokenizer.decode(output)
            #     # response = re.sub(r'\n ', '', response)
            #     response = response.replace("<pad>", "").replace("</s>", "").replace("<s>", "").replace("<unk>", "").replace(" ", "").replace("\n", "")
            #     result.append(response)
        # for re in results:
        #     print("RESULTS: ", re)
        # exit()
        # print('>>>results before postprocess:',results,'\nresults unpostprocessed end<<<')
        results = [postprocess(result,self.name) for result in results]
        # print('>>>results after postprocess:',results,'\nresults unpostprocessed end<<<')
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
        if dataset_name in ['MMLU','RAFT','C-Eval']:
            results = [postprocess_(result,self.name) for result in results]
        else:
            results = [postprocess(result,self.name) for result in results]
        return results

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
}
