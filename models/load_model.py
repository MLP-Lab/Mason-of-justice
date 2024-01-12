import peft 
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model

kwargs = {
    #"device_map" : "auto",
    "torch_dtype" : torch.float16,
    "attn_implementation": "flash_attention_2",
}

def load_model(model_name= "meta-llama/Llama-2-7b-hf"):

    print("========= Load Model =========")
    print(model_name)
    llama_config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=llama_config, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"# of model parameters (Before LoRA): {model.num_parameters()}")

    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        modules_to_save=['embed_tokens','lm_head']
    )
    model = get_peft_model(model, lora_config)
    print(f"# of model parameters (After LoRA) : {model.num_parameters()}")
    model.print_trainable_parameters()

    print(f"Model dtype : {model.dtype}")

    return model, tokenizer