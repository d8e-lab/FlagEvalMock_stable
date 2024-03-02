import re
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, LlamaTokenizerFast

def preprocess(query):
    query=re.sub(r"^[\s\n\t\ufff0-\uffff]+", "", query)
    query=re.sub(r"[\s\n\t\ufff0-\uffff]+$", "", query)
    return query

class BaseLLM:
    def __init__(self, model_name, model_path, tokenizer_path, gpu_id=0) -> None:
        self.name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path ,padding_side='left',truncation_side="left" , trust_remote_code=True
        )
        print('gpu_id:',gpu_id)            
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).bfloat16().to(gpu_id).eval()
        )

    def inference(self, queries, nt):
        gen_kwargs = {
            "max_new_tokens": nt,
            "num_beams": 1,
            "do_sample": False,
            # "top_p": 0.9,
            # "temperature": 0.1,
        } 
        # 去除token_type_ids
        queries = [preprocess(q) for q in queries]
        inputs = {key:value for key,value in inputs.items() if key!='token_type_ids'}
        outputs = self.model.generate(pad_token_id=self.tokenizer.pad_token_id,**inputs, **gen_kwargs)
        outputs = outputs[:,len(inputs["input_ids"][0]):]
        results = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
            
        return results
