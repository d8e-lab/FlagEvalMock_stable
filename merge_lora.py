import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
 
 
def apply_lora(model_name_or_path, output_path, lora_path):
    print(f"Loading the base model from {model_name_or_path}")
    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    base_tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
 
    print(f"Loading the LoRA adapter from {lora_path}")
 
    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.float16,
    )
 
    print("Applying the LoRA")
    model = lora_model.merge_and_unload()
 
    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)



model_name_or_path = "/mnt/40_store/SiYuan/chat_sftv3-9750/"
#'/mnt/store/llama2-checkpoints-plus-continue/checkpoint-2000'
output_path = "/mnt/40_store/xxw/trl/ppo_saved/siyuan_hh_hfrl_rm_1203/step_59/merge"
#'/mnt/SFT_store/xxw/outputs/all_peft/lora_continue/2023-09-18_04-02-42_merge'
lora_path = "/mnt/40_store/xxw/trl/ppo_saved/siyuan_hh_hfrl_rm_1203/step_59"
#'/mnt/SFT_store/xxw/outputs/all_peft/lora_continue/2023-09-18_04-02-42'

apply_lora(model_name_or_path, output_path, lora_path)
