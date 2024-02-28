import torch
from copy import deepcopy

ckpt = torch.load("/data/megatron_ckpt/raw_model/stage2/stage2_final_parallel1/iter_0021000/mp_rank_00/model_optim_rng.pt")
pruned_ckpt_path = "/data/megatron_ckpt/raw_model/8-24/model_parallel1/iter_0021000/mp_rank_00/model_optim_rng.pt"

pruned_dec = [7,14,21]
decoder_number = [i for i in range(40) if i not in pruned_dec]

print(decoder_number)

def get_prune_layers(ckpt,assinged_layers):
    layers = {}


    for i in range(len(assinged_layers)):
        old_prefix = f"layers.{assinged_layers[i]}."
        new_prefix = f"layers.{i}."
        print(new_prefix,old_prefix)
        # import pdb
        # pdb.set_trace()
        for key, value in ckpt.items():
            if old_prefix in key:
                new_key = new_prefix + key[len(old_prefix):]
                layers[new_key] = value
    for key, value in ckpt.items():
        if "layers." not in key:
            layers[key] = value
    return layers


new_ckpt = deepcopy(ckpt)

new_encoder = get_prune_layers(ckpt["model"]["language_model"]['encoder'],encoder_number)

new_ckpt["model"]["language_model"]['encoder'] = new_encoder

new_decoder = get_prune_layers(ckpt["model"]["language_model"]['decoder'],decoder_number)

new_ckpt["model"]["language_model"]['decoder'] = new_decoder

new_ckpt["args"].num_layers = len(encoder_number)
new_ckpt["args"].encoder_num_layers = len(encoder_number)
new_ckpt["args"].decoder_num_layers = len(decoder_number)

torch.save(new_ckpt,pruned_ckpt_path)