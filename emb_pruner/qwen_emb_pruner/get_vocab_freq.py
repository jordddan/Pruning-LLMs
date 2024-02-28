import sys
import os
import sentencepiece as spm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import argparse

from tqdm import tqdm
import torch
import tiktoken
from transformers import AutoTokenizer
'''
    This python script can generate the ids of the remaining vocab 
    according to the based on frequency of each token in the target file.
'''
def main(args):

    # cl100k_base = tiktoken.get_encoding(args.tkm)
    tokenizer = AutoTokenizer.from_pretrained("/nvme/hf_models/qwen/Qwen-1_8B", trust_remote_code=True)
    freq = {}
    for i in range(151643):
        freq[i] = 0
    # import pdb
    # pdb.set_trace()
    # count the vocab occurrence frequency in target corpus
    if args.data_type == "txt":
        with open(args.data_path,'r') as f:
            data = f.read()

        sep_data = []
        sep_len = 1024
        left = 0

        while left < len(data)-1:
            sep_data.append(data[left:left+sep_len])
            left += sep_len
        
        for line in tqdm(sep_data):
            line_ids = tokenizer.encode(line)
            for id in line_ids:
                if id < 151643:
                    freq[id] += 1


    sort_value = [(key, value) for key,value in freq.items()]
    sort_value.sort(reverse=True, key=lambda x:x[1])

    # save the vocab file sorted by the vocab frequency
    with open(f"{args.vocab_freq}.txt", 'w', encoding='utf-8') as f:
        for key, value in sort_value:
            f.write("{} {}\n".format(key, value))
    print("generate vocab freq file finish !!!")

    # save the tensor of the remained vocab_ids
    remain_ids = [item[0] for item in sort_value[:args.remain_vocab_size]]
    torch.save(remain_ids,args.remain_vocab_ids_path)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type",type=str,default="txt")
    parser.add_argument("--data_path",type=str,default="/home/amax/qd/LEO_V2_DEV/LEO_debug/emb_pruner/llama_emb_pruner/mmlu.json")
    parser.add_argument("--vocab_freq",type=str,help="the prefix name for the vocab freq file", default="vocab_freq.txt")
    parser.add_argument("--model_name_or_path",type=str,default="/nvme/hf_models/qwen/Qwen-1_8B",help="path of the pretrained model including tiktoken model")
    parser.add_argument("--remain_vocab_size",type=int,default=30000)
    parser.add_argument("--remain_vocab_ids_path",type=str, default="remain_vocab_ids.pt")
    args = parser.parse_args()
    
    main(args)