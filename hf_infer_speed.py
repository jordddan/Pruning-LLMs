import os
import time
from transformers import AutoTokenizer, BitsAndBytesConfig
import torch
from modeling_llama import LlamaForCausalLM

# Load the tokenizer and model
tokenizer_path = "/data/hf_models/Baichuan2-7B-Base"
model_path = "/data/Megatron-LM-main/ckpts/130B_hf"


tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
# quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    device_map="auto", 
    load_in_4bit=True)

# Function to generate text
def generate(st, max_new_tokens):
    model_inputs = tokenizer(st, return_tensors="pt").to("cuda")
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    print(generated_ids.shape)
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Test token generation speed
def test_token_generation_speed(start_text, num_tokens_to_generate, num_trials=5):
    times = []
    for _ in range(num_trials):
        start_time = time.time()
        generate(start_text, num_tokens_to_generate)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / num_trials
    return avg_time

# Example usage
start_text = "This code is based"
num_tokens_to_generate = 200  # Number of tokens to generate in each trial
num_trials = 5  # Number of trials to average over

avg_generation_time = test_token_generation_speed(start_text, num_tokens_to_generate, num_trials)
print(f"Average generation time for {num_tokens_to_generate} tokens over {num_trials} trials: {avg_generation_time} seconds")
