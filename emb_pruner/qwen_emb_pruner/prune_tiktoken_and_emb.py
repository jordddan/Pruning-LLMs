import sys
import os
import sentencepiece as spm
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import torch
import json
import argparse
import base64
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open

def load_tiktoken_bpe(tiktoken_bpe_file: str) -> "dict[bytes, int]":
    contents = open(tiktoken_bpe_file, "rb").read()
    return {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in contents.splitlines() if line)
    }

def dump_tiktoken_bpe(bpe_ranks: "dict[bytes, int]", tiktoken_bpe_file: str) -> None:
    with open(tiktoken_bpe_file, "wb") as f:
        for token, rank in sorted(bpe_ranks.items(), key=lambda x: x[1]):
            f.write(base64.b64encode(token) + b" " + str(rank).encode() + b"\n")

def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device='cpu') as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
        metadata = f.metadata()
    # save_file(tensors, "model.safetensors")
    return tensors, metadata

def create_new_tiktoken_by_idx(args, remain_vocab_ids:list):
    
    '''
    intro:
        1. first add special token to the remain_vocab_ids 
        2. create new sentencepiece model according to the remain_vocab_ids
    args:
        remain_vocab_ids: the ids of the remaining tokens
    '''
    tiktoken_path = os.path.join(args.hf_dir,"qwen.tiktoken")
    mergeable_ranks = load_tiktoken_bpe(tiktoken_path)

    remain_ids_with_spt = remain_vocab_ids + [i for i in range(151643,151936)]
    remain_vocab_ids_set = set(remain_vocab_ids)
    tokens_remain = []
    for k,v in mergeable_ranks.items():
        if v in remain_vocab_ids_set:
            tokens_remain.append((k,v))
    tokens_remain = sorted(tokens_remain,key=lambda x:x[1])
    new_ranks = {}
    
    for i in range(len(tokens_remain)):
        new_ranks[tokens_remain[i][0]] = i

    new_tiktoken_path = os.path.join(args.hf_dir, "qwen.tiktoken")
    dump_tiktoken_bpe(new_ranks, new_tiktoken_path)

    torch.save(remain_ids_with_spt,"remain_ids_with_spt.pt")
    return remain_ids_with_spt


def prune_qwen_by_ids(hf_dir,remain_ids):
    '''
    intro:
        first find the weight in model_index file 
        llama embedding weight and the lm_head weight and overwrite original file

    args:
        hf_dir: the huggingface model path 

    '''
    pytorch_model_index_config_path = os.path.join(hf_dir, "model.safetensors.index.json")
    with open(pytorch_model_index_config_path,'r') as f:
        pytorch_model_index_config = json.load(f)

    lm_head_path = pytorch_model_index_config["weight_map"]["lm_head.weight"]
    emb_path = pytorch_model_index_config["weight_map"]["transformer.wte.weight"]

    lm_head_path = os.path.join(hf_dir,lm_head_path)
    emb_path = os.path.join(hf_dir,emb_path)

    # prune lm_head and save the new ckpt to the original weight path
    total_weight, metadata = load_safetensors(lm_head_path)
    lm_head_weight = total_weight["lm_head.weight"] # [vocab_size,hidden_size]

    pruned_lm_head_weight = lm_head_weight.index_select(0,remain_ids)

    total_weight["lm_head.weight"] = pruned_lm_head_weight
    # import pdb
    # pdb.set_trace()
    save_file(total_weight,lm_head_path,metadata=metadata)


    # prune emb
    total_weight, metadata = load_safetensors(emb_path)
    emb_weigth = total_weight["transformer.wte.weight"] # [vocab_size,hidden_size]
    pruned_emb_weight = emb_weigth.index_select(0,remain_ids)
    total_weight["transformer.wte.weight"] = pruned_emb_weight


    vocab_size = remain_ids.shape[0]
    print(f"The final vocab size with special tokens is {vocab_size}")
    # import pdb
    # pdb.set_trace()
    save_file(total_weight,emb_path, metadata=metadata)

    config_path = os.path.join(hf_dir,"config.json")
    with open(config_path,'r') as f:
        config = json.load(f)
    
    config["vocab_size"] = vocab_size
    with open(config_path,'w') as f:
        json.dump(config,f,indent=1)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remain_vocab_path",type=str,default="remain_vocab_ids.pt")
    parser.add_argument("--hf_dir",type=str)
    parser.add_argument("--remain_ids_with_spt",type=str)
    args = parser.parse_args()
    remain_vocab_ids = torch.load(args.remain_vocab_path)

    remain_ids_with_spt = create_new_tiktoken_by_idx(args, remain_vocab_ids)

    prune_qwen_by_ids(args.hf_dir, torch.LongTensor(remain_ids_with_spt))

    # reassign special tokens
    generation_config_path = os.path.join(args.hf_dir,"generation_config.json")
    with open(generation_config_path,'r') as f:
        generation_config = json.load(f)
    
    generation_config["eos_token_id"] = len(remain_vocab_ids)
    generation_config["pad_token_id"] = len(remain_vocab_ids)
    generation_config["stop_words_ids"] = [[len(remain_vocab_ids)]]
    with open(generation_config_path,'w') as f:
        json.dump(generation_config,f,indent=1)