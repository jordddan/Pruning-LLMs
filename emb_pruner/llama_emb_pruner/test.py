from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM

import torch
from transformers import AutoTokenizer
import os

# load base LLM model and tokenizer
model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path="/nvme/qd/ckpt/prune_llama2",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    local_files_only=True,
    use_safetensors=False
)

# import pdb
# pdb.set_trace()

tokenizer = AutoTokenizer.from_pretrained("/nvme/qd/ckpt/prune_llama2")

prompt = f"""What is the color of the sky?"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(model.device)
outputs = model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9,temperature=0.9)

print(f"Generated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")