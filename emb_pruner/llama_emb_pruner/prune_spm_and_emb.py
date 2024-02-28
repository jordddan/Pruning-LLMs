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

def create_new_spm_by_idx(remain_vocab_ids, spm_path):
    
    '''
    intro:
        1. first add special token to the remain_vocab_ids 
        2. create new sentencepiece model according to the remain_vocab_ids
    args:
        remain_vocab_ids: the ids of the remaining tokens
        spm_path: sentencepiece model path 
    '''
    remain_vocab_ids = remain_vocab_ids
    sp_model = spm.SentencePieceProcessor(model_file=spm_path)
    remain_tokens = set()

    for token_id in remain_vocab_ids:
        remain_tokens.add(sp_model.IdToPiece(token_id))

    old_spm = sp_pb2_model.ModelProto()

    old_spm.ParseFromString(open(spm_path, "rb").read())

    new_sp_model = sp_pb2_model.ModelProto()

    new_sp_model.ParseFromString(open(spm_path, "rb").read())
    new_sp_model.pieces.clear()
    token_id = 0
    remain_tokens_with_spt = []
    for item in old_spm.pieces:
        if item.piece in remain_tokens:
            new_sp_model.pieces.append(item)
            remain_tokens_with_spt.append(item.piece)
        elif item.type != 1:
            new_sp_model.pieces.append(item)
            remain_tokens_with_spt.append(item.piece)
    print(len(remain_tokens_with_spt))

    old_spm = spm.SentencePieceProcessor(spm_path)
    remain_ids_with_spt = [old_spm.PieceToId(token) for token in remain_tokens_with_spt]
    remain_ids_with_spt = torch.LongTensor(remain_ids_with_spt)
    return new_sp_model, remain_ids_with_spt


def prune_llama_by_ids(hf_dir,remain_ids):
    '''
    intro:
        first find the weight in model_index file 
        llama embedding weight and the lm_head weight and overwrite original file

    args:
        hf_dir: the huggingface model path 

    '''
    pytorch_model_index_config_path = os.path.join(hf_dir, "pytorch_model.bin.index.json")
    with open(pytorch_model_index_config_path,'r') as f:
        pytorch_model_index_config = json.load(f)

    lm_head_path = pytorch_model_index_config["weight_map"]["lm_head.weight"]
    emb_path = pytorch_model_index_config["weight_map"]["model.embed_tokens.weight"]

    lm_head_path = os.path.join(hf_dir,lm_head_path)
    emb_path = os.path.join(hf_dir,emb_path)

    # prune lm_head and save the new ckpt to the original weight path
    total_weight = torch.load(lm_head_path)
    lm_head_weight = total_weight["lm_head.weight"] # [vocab_size,hidden_size]
    pruned_lm_head_weight = lm_head_weight.index_select(0,remain_ids)

    total_weight["lm_head.weight"] = pruned_lm_head_weight
    # import pdb
    # pdb.set_trace()
    torch.save(total_weight,lm_head_path)


    # prune emb
    total_weight = torch.load(emb_path)
    emb_weigth = total_weight["model.embed_tokens.weight"] # [vocab_size,hidden_size]
    pruned_emb_weight = emb_weigth.index_select(0,remain_ids)
    total_weight["model.embed_tokens.weight"] = pruned_emb_weight


    vocab_size = remain_ids.shape[0]
    print(f"The final vocab size with special tokens is {vocab_size}")
    # import pdb
    # pdb.set_trace()
    torch.save(total_weight,emb_path)

    config_path = os.path.join(hf_dir,"config.json")
    with open(config_path,'r') as f:
        config = json.load(f)
    
    config["vocab_size"] = vocab_size
    with open(config_path,'w') as f:
        json.dump(config,f,indent=1)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remain_vocab_path",type=str,default="mmap")
    parser.add_argument("--hf_dir",type=str)
    parser.add_argument("--remain_ids_with_spt",type=str)
    args = parser.parse_args()
    remain_vocab_ids = torch.load(args.remain_vocab_path)
    spm_path = os.path.join(args.hf_dir,"tokenizer.model")
    # import pdb
    # pdb.set_trace()
    new_sp_model, remain_ids_with_spt = create_new_spm_by_idx(remain_vocab_ids,spm_path)
    torch.save(remain_ids_with_spt,args.remain_ids_with_spt)

    with open(spm_path, 'wb') as f:
        f.write(new_sp_model.SerializeToString())
    prune_llama_by_ids(args.hf_dir,remain_ids_with_spt)


