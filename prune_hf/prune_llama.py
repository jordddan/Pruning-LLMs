from modeling_llama import LlamaForCausalLM
from transformers import AutoConfig
from transformers import Trainer
from transformers import LlamaConfig
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path",type=str)
parser.add_argument("--prune_config_path",type=str,default="prune_config.json")
parser.add_argument("--output_path",type=str)
args = parser.parse_args()


llama_config = AutoConfig.from_pretrained(args.model_name_or_path)
with open(args.prune_config_path) as f:
    prune_config = json.load(f)
llama_config.update(prune_config)


def compute_model_size(model):
    sum = 0
    for name, params in model.named_parameters():
        temp = 1
        for j in params.shape:
            temp *= j
    sum += temp 
    return sum / 1e9

llama_model = LlamaForCausalLM(llama_config).cuda()
llama_model.eval()

original_model_size = compute_model_size(llama_model)
print(f"Original Model Size: {original_model_size}B")


new_config = AutoConfig.from_pretrained(args.model_name_or_path)
new_config.hidden_size = llama_model.config.hidden_size_remain
new_config.intermediate_size = llama_model.config.ffn_hidden_size_remain
new_config.num_attention_heads = llama_model.config.num_attention_heads_remain
new_config.num_hidden_layers = len(llama_model.model.layers)

llama_model.prune()
original_model_size = compute_model_size(llama_model)
print(f"Pruned Model Size: {original_model_size}B")

llama_model.config = new_config

trainer = Trainer(model=llama_model)
trainer._save(args.output_path)