import sys
import os
import sentencepiece as spm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from  megatron.data.dataset_utils import *

import argparse

from tqdm import tqdm
import torch

'''
    This python script can generate the ids of the remaining vocab 
    according to the based on frequency of each token in the target file.
'''
def main(args):

    sp_model = spm.SentencePieceProcessor(model_file=args.spm)
    freq = {}
    vocab_size = sp_model.vocab_size()
    print(vocab_size)
    for i in range(vocab_size):
        piece = sp_model.IdToPiece(i)
        freq[i] = 0


    # count the vocab occurrence frequency in target corpus
    if args.data_type == "txt":
        with open(args.data_path,'r') as f:
            data = f.readlines()
        for line in tqdm(data):
            line_ids = sp_model.Encode(line)
            for id in line_ids:
                # piece = sp_model.IdToPiece(id)
                freq[id] += 1

    elif args.data_type == "mmap":  
        index_dataset = get_indexed_dataset_(args.data_path, 'mmap', True)
        for i in tqdm(range(0, len(index_dataset)+1, 2)):
            line = index_dataset.get(i)
            for id in line:
                # piece = sp_model.IdToPiece(id)
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
    parser.add_argument("--data_type",type=str,default="mmap")
    parser.add_argument("--data_path",type=str,default="/home/amax/qd/LEO_V2_DEV/LEO_debug/emb_pruner/mmlu.json")
    parser.add_argument("--vocab_freq",type=str,help="the prefix name for the vocab freq file")
    parser.add_argument("--model_name",type=str,default="llama")
    parser.add_argument("--spm",type=str,default="/nvme/hf_models/Llama-2-7b-hf/tokenizer.model",help="path of the sentencepiece model")
    parser.add_argument("--remain_vocab_size",type=int,default=20000)
    parser.add_argument("--remain_vocab_ids_path",type=str)
    parser.add_argument("--new_spm_path",type=str)
    args = parser.parse_args()
    
    main(args)