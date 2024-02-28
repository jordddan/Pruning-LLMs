import torch
ckpt1 = torch.load("/data/megatron_ckpt/test_pruned_7B_tp1/iter_0001000/mp_rank_00/model_optim_rng.pt",map_location=torch.device("cpu"))
ckpt2 = torch.load("/data/megatron_ckpt/pruned_7B-hf/1000/pytorch_model-00001-of-00003.bin",map_location=torch.device("cpu"))
print(ckpt1["model"]["language_model"]["embedding"]["word_embeddings"]["weight"].sum())
print(ckpt2["model.embed_tokens.weight"].sum())
import pdb
pdb.set_trace()