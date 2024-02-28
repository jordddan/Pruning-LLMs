# import torch
# import torch.distributed as dist
# import os
# print(os.environ)
# print("|| MASTER_ADDR:",os.environ["MASTER_ADDR"],
# "|| MASTER_PORT:",os.environ["MASTER_PORT"],
# "|| LOCAL_RANK:",os.environ["LOCAL_RANK"],
# "|| RANK:",os.environ["RANK"], 
# "|| WORLD_SIZE:",os.environ["WORLD_SIZE"])
# print()
# dist.init_process_group('nccl', rank=int(os.environ["RANK"]), 
# world_size=int(os.environ["WORLD_SIZE"]))
# print("done")
# tensor = torch.ones(1).cuda(int(os.environ["LOCAL_RANK"]))
# print("init done")
# dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
# print(tensor)
# dist.destroy_process_group()
import torch
ckpt1 = torch.load("/data/megatron_ckpt/test_pruned_7B_tp1/iter_0001000/mp_rank_00/model_optim_rng.pt",map_location=torch.device("cpu"))
ckpt2 = torch.load("/data/megatron_ckpt/pruned_7B-hf/1000/pytorch_model-00001-of-00003.bin",map_location=torch.device("cpu"))
print(ckpt1["model"]["language_model"]["embedding"]["word_embeddings"]["weight"].sum())
print(ckpt2["model.embed_tokens.weight"].sum())
import pdb
pdb.set_trace()