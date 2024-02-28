import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm

# openba_sp_model_file = "/home/amax/qd/MODEL_CKPT/hf_models/OpenBT5-Flan-Pruned/spiece.model"
openba_sp_model_file = "/home/amax/qd/emb_prune/tokenizer/spiece.model"
vocab_file_path = "/home/amax/qd/emb_prune/new_vocab.txt"



openba_spm_model = sp_pb2_model.ModelProto()
with open(vocab_file_path,"r") as f:
    data = f.readlines()
tokens = []
for token in data:
    tokens.append(token.strip("\n"))

openba_spm_model = sp_pb2_model.ModelProto()
openba_spm_model.ParseFromString(open(openba_sp_model_file, "rb").read())

token_dict = set(tokens)


new_sp_model = sp_pb2_model.ModelProto()

new_sp_model.ParseFromString(open(openba_sp_model_file, "rb").read())
new_sp_model.pieces.clear()

for item in openba_spm_model.pieces:
    if item.piece in token_dict:
        new_sp_model.pieces.append(item)
    elif item.type != 1:
        new_sp_model.pieces.append(item)
print(len(new_sp_model.pieces))


with open('newspiece2.model', 'wb') as f:
    f.write(new_sp_model.SerializeToString())