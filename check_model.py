import torch
# zs = ckpt2["model"]["zs"].cuda()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
ckpt1 = torch.load("/data/LEO/megatron_prune/ckpts/Llama2-13b_mp8/iter_0000001/mp_rank_00/model_optim_rng.pt")
ckpt2 = torch.load("/data/megatron_ckpt/llama13B_pruned_7B_tp8/iter_0000001/mp_rank_00/model_optim_rng.pt")
import pdb
pdb.set_trace()
zs = ckpt2["model"]["zs"]

z_hidden = zs["hidden_index"].cuda()
inter_z = zs["intermediate_indexes"][5].cuda()
model1 = ckpt1["model"]["language_model"]
model2 = ckpt2["model"]["language_model"]


w1 = model1["output_layer"]["weight"]
w2 = model2["output_layer"]["weight"]

embedding1 = model1["embedding"]["word_embeddings"]["weight"]
embedding2 = model2["embedding"]["word_embeddings"]["weight"]


layer_norm_weight1 = model1["encoder"]["layers.5.input_norm.weight"]
layer_norm_weight2 = model2["encoder"]["layers.3.input_norm.weight"]
mlp_weight1 = model1["encoder"]["layers.5.mlp.dense_h_to_4h.weight"]
mlp_weight2 = model2["encoder"]["layers.3.mlp.dense_h_to_4h.weight"]

expanded_inter_z = torch.cat([inter_z,inter_z+mlp_weight1.shape[0]//2])

# print(mlp_weight1.index_select(-1,z_hidden).index_select(0,expanded_inter_z).sum())
# print(mlp_weight2.sum())
import pdb
pdb.set_trace()

