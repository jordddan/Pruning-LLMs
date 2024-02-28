# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import shutil
import json
import os
import re
import sys
import types

import torch

from transformers import LlamaConfig
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint


def add_checkpointing_args(parser):
    parser.add_argument("--megatron-path", type=str, default=None, help="Base directory of Megatron repository")
    parser.add_argument(
        "--convert_checkpoint_from_megatron_to_transformers",
        action="store_true",
        help=(
            "If True, convert a Megatron checkpoint to a Transformers checkpoint. "
            "If False, convert a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--load_path",
        type=str,
        required=True,
        help="Path to the checkpoint to convert.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to the converted checkpoint.",
    )
    parser.add_argument("--print-checkpoint-structure", action="store_true")

    return parser


def add_megatron_checkpoint_args(parser):
    parser.add_argument(
        "--target_tensor_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The tensor model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_pipeline_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The pipeline model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_data_parallel_size",
        type=int,
        default=1,
        help=(
            "The data parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_params_dtype",
        type=str,
        default="fp32",
        help=(
            "The dtype of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--make_vocab_size_divisible_by",
        type=int,
        default=128,
        help=(
            "Pad the vocab size to be divisible by this value. "
            "This is added for computational efficieny reasons. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--use_distributed_optimizer",
        action="store_true",
        help=(
            "If True, use the distributed optimizer. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    return parser


def add_transformers_checkpoint_args(parser):
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help=(
            "The name of the pre-trained tokenizer to save. "
            "If not None, the tokenizer will be saved. "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="10GB",
        help=(
            "The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size "
            "lower than this size. If expressed as a string, needs to be digits followed by a unit (like `5MB`). "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )

    return parser


# The simple map of names for "automated" rules.
megatron_to_transformers = {
    "self_attention.dense": ".self_attn.o_proj.",
    "mlp.dense_4h_to_h": ".mlp.down_proj.",
}

tensor_parallel_params = [
    # megatron-lm layers to merge across tp ranks
    "self_attention.query_key_value.weight",
    "self_attention.query_key_value.bias",
    "self_attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
    "mlp.dense_4h_to_h.weight",
    # deprecated
    "attention.query_key_value.weight",
    "attention.query_key_value.bias",
    "attention.dense.weight",
    # transformers layers to split across tp ranks
    "attn.c_attn.weight",
    "attn.c_attn.bias",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_fc.bias",
    "mlp.c_proj.weight",
    'self_attn.q_proj.weight',
    'self_attn.k_proj.weight',
    'self_attn.v_proj.weight',
    'self_attn.o_proj.weight',
    'mlp.down_proj.weight',
    'mlp.up_proj.weight',
    'mlp.gate_proj.weight'
]


def recursive_print(name, val, spaces=0):
    """
    Recursively print the structure of a checkpoint. This function is taken from `convert_megatron_gpt2_checkpoint.py`

    Args:
        name (str): the name of the current tensor parameter
        val (Tuple(int)): the shape of the current tensor parameter
        spaces (int): the number of spaces to print before the output for a nested structure
    """
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def merge_transformers_sharded_states(path, num_checkpoints):
    """
    Merge sharded checkpoints from transformers into a single checkpoint.

    Args:
        path (str): the path to the sharded checkpoints
        num_checkpoints (int): the number of checkpoints to merge
    """
    state_dict = {}
    for i in range(1, num_checkpoints + 1):
        print('loading', i, ':', num_checkpoints + 1)
        checkpoint_path = os.path.join(path, f"pytorch_model-{i:05d}-of-{num_checkpoints:05d}.bin")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(path, f"pytorch_model-{i}-of-{num_checkpoints}.bin")
            assert os.path.exists(checkpoint_path), f"Cannot find checkpoint {checkpoint_path}"
        current_chunk = torch.load(checkpoint_path, map_location="cpu")
        state_dict.update(current_chunk)
    return state_dict


def get_megatron_sharded_states(load_path, tp_size, pp_size, pp_rank):
    """
    Get sharded checkpoints from NVIDIA Megatron-LM checkpoint based on the provided tensor parallel size, pipeline
    parallel size and pipeline parallel rank.

    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts = []
    for i in range(tp_size):
        possible_sub_dir_names = [
            f"mp_rank_{i:02d}" if pp_size == 1 else f"mp_rank_{i:02d}_{pp_rank:03d}",
            f"mp_rank_{i:02d}_dp_000" if pp_size == 1 else f"mp_rank_{i:02d}_{pp_rank:03d}_dp_000"
        ]
        sub_dir_name = None
        for p in possible_sub_dir_names:
            if os.path.exists(os.path.join(load_path, p)):
                sub_dir_name = p
                break
        assert sub_dir_name is not None, f"Cannot find sub dir in {possible_sub_dir_names}"
        checkpoint_path = os.path.join(load_path, sub_dir_name, 'model_optim_rng.pt')
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        tp_state_dicts.append(state_dict)
    return tp_state_dicts


def get_element_from_dict_by_path(d, path):
    """
    Get element from dictionary by path. If element is not present, recursively add empty dictionaries.

    Args:
        d (dict): the dictionary to get the element from
        path (list): the path to the element which is delimited by "."
    """
    path = path.split(".")
    for k in path:
        if k not in d:
            d[k] = {}
        d = d[k]
    return d


def copy_tokenizer(args):
    os.makedirs(args.save_path, exist_ok=True)
    tokenizer_dir = args.load_path
    if os.path.exists(os.path.join(args.load_path, 'tokenizer')):
        tokenizer_dir = os.path.join(args.load_path, 'tokenizer')
    file_list = os.listdir(tokenizer_dir)
    for f in file_list:
        if 'token' in f:
            shutil.copyfile(os.path.join(tokenizer_dir, f), os.path.join(args.save_path, f))


def convert_checkpoint_from_megatron_to_transformers(args):
    """
    Convert NVIDIA Megatron-LM checkpoint to HuggingFace Transformers checkpoint. This handles Megatron checkpoints
    with different tensor parallelism and pipeline parallelism sizes. It saves the converted checkpoint into shards
    using HuggingFace Transformers checkpoint sharding functionality. 

    Args:
        args (argparse.Namespace): the arguments to the script
    """
    # Search in directory above this
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    # Load Megatron-LM checkpoint arguments from the state dict
    sub_dirs = os.listdir(args.load_path)
    release = False
    if 'latest_checkpointed_iteration.txt' in sub_dirs:
        with open(os.path.join(args.load_path, 'latest_checkpointed_iteration.txt')) as f:
            latest_ckpt = f.readline().strip()
            print(f"latest checkpoint: {latest_ckpt}")
            if isinstance(latest_ckpt, bytearray):
                latest_ckpt = latest_ckpt.decode("utf-8")
            try:
                iteration = int(latest_ckpt)
            except ValueError:
                release = (latest_ckpt == "release")
                if not release:
                    raise ValueError(f"Invalid latest checkpoint: {latest_ckpt}")
            for sub_dir in sub_dirs:
                if latest_ckpt in sub_dir:
                    latest_ckpt = sub_dir
                    break
    else:
        raise ValueError('Cannot find latest ckpt!')
    possible_state_paths = [os.path.join(args.load_path, latest_ckpt),
                  os.path.join(args.load_path, latest_ckpt, 'iter_'+str(iteration) if not release else 'release')]
    state_path = None
    for p in possible_state_paths:
        if os.path.exists(p):
            state_path = p
            print(f"Loading Megatron-LM checkpoint arguments from: {state_path}")
            break
    assert state_path is not None, f"Cannot find state path in {possible_state_paths}"
    possible_sub_dirs = ["mp_rank_00", "mp_rank_00_000", "mp_rank_00_dp_000", "mp_rank_00_000_dp_000"]
    state_dirs = os.listdir(state_path)
    for sub_dir in possible_sub_dirs:
        if sub_dir in state_dirs:
            rank0_checkpoint_path = os.path.join(state_path, sub_dir, 'model_optim_rng.pt')
            break
    print(f"Loading Megatron-LM checkpoint arguments from: {rank0_checkpoint_path}")
    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")
    megatron_args = state_dict.get("args", None)
    if megatron_args is None:
        raise ValueError(
            "Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints"
            " containing all the megatron arguments. This is because it loads all config related to model"
            " architecture, the tensor and pipeline model parallel size from the checkpoint insead of user having to"
            " manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron"
            " arguments to use this utility."
        )

    # Create Transformers GPT2 config from Megatron-LM arguments
    vocab_size = megatron_args.padded_vocab_size

    # params dtype
    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    config = LlamaConfig(
        bos_token_id=1,
        eos_token_id=2,
        hidden_act='silu',
        hidden_size=megatron_args.hidden_size,
        intermediate_size=megatron_args.ffn_hidden_size,
        initializer_range=0.02,
        max_sequence_length=megatron_args.seq_length,
        model_type='llama',
        num_attention_heads=megatron_args.num_attention_heads,
        num_hidden_layers=megatron_args.num_layers,
        pad_token_id=0,
        rms_norm_eps=megatron_args.norm_epsilon,
        torch_dtype='bfloat16',
        transformers_version='4.28.0.dev0',
        use_cache=True,
        vocab_size=vocab_size,
        architectures=["LLaMAForCausalLM"],
    )

    output_state_dict = {}

    tp_size = megatron_args.tensor_model_parallel_size
    pp_size = megatron_args.pipeline_model_parallel_size
    
    # The regex to extract layer names.
    layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # Convert.
    print("Converting")

    # Embeddings
    print("Converting embeddings")
    tp_state_dicts = get_megatron_sharded_states(state_path, tp_size, pp_size, 0)

    # Convert and store the word embeddings.
    word_embeddings = torch.cat(
        [
            get_element_from_dict_by_path(
                tp_state_dicts[tp_rank], "model.language_model.embedding.word_embeddings.weight"
            )
            for tp_rank in range(tp_size)
        ],
        dim=0,
    )
    word_embeddings = word_embeddings[:vocab_size].to(dtype).clone().detach().contiguous()
    output_state_dict["model.embed_tokens.weight"] = word_embeddings

    # Transformer Layers
    print("Converting transformer layers")
    # The hidden_size per head.
    hidden_size_per_head = config.hidden_size // config.num_attention_heads
    num_layers = config.num_hidden_layers // pp_size

    for pp_rank in range(pp_size):
        if pp_size > 0:
            print(f"Converting pipeline parallel rank {pp_rank}")
            tp_state_dicts = get_megatron_sharded_states(state_path, tp_size, pp_size, pp_rank)

        # The transformer.
        path = "model.language_model.encoder"

        # Extract the layers.
        for key, val in get_element_from_dict_by_path(tp_state_dicts[0], path).items():
            # Match the name.
            m = layer_re.match(key)
            # Stop if that's not a layer
            if m is None:
                break

            # The index of the layer.
            layer_idx = int(m.group(1)) + pp_rank * num_layers
            # The name of the operation.
            op_name = m.group(2)
            # Is it a weight or a bias?
            weight_or_bias = m.group(3)

            # The name of the layer.
            layer_name = f"model.layers.{layer_idx}"

            if op_name + "." + weight_or_bias not in tensor_parallel_params:
                params = val.to(dtype)
            else:
                dim = 1 if op_name in ["self_attention.dense", "mlp.dense_4h_to_h"] else 0
                params = torch.cat(
                    [val]
                    + [
                        get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                        for tp_rank in range(1, tp_size)
                    ],
                    dim=dim,
                ).to(dtype)

            # For layernorm(s), simply store the layer norm.
            if op_name.endswith("norm"):
                ln_name = "input_layernorm" if op_name.startswith("input") else "post_attention_layernorm"
                output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = params

            # Split QKV packed weights
            elif op_name == "self_attention.query_key_value" and weight_or_bias == "weight":
                params_per_tp = params.chunk(dim=0, chunks=megatron_args.tensor_model_parallel_size)
                
                q = torch.empty(0)
                k = torch.empty(0)
                v = torch.empty(0)
                for t in params_per_tp:
                    attn_heads = t.chunk(3 * state_dict['args'].num_attention_heads)
                    qp, kp, vp = [], [], []
                    for idx, w in enumerate(attn_heads):
                        if idx % 3 == 0:
                            qp.append(w)
                        elif idx % 3 == 1:
                            kp.append(w)
                        else:
                            vp.append(w)
                    qp, kp, vp = torch.cat(qp, dim = 0), torch.cat(kp, dim = 0), torch.cat(vp, dim = 0), 
                    q = torch.cat([q, qp])
                    k = torch.cat([k, kp])
                    v = torch.cat([v, vp])
                output_state_dict[layer_name + ".self_attn.q_proj.weight"] = q.to(dtype).clone().detach().contiguous()
                output_state_dict[layer_name + ".self_attn.k_proj.weight"] = k.to(dtype).clone().detach().contiguous()
                output_state_dict[layer_name + ".self_attn.v_proj.weight"] = v.to(dtype).clone().detach().contiguous()

            elif op_name == "mlp.dense_h_to_4h" and weight_or_bias == "weight":
                params_per_tp = params.chunk(dim=0, chunks=megatron_args.tensor_model_parallel_size)
                gate = torch.empty(0)
                up = torch.empty(0)
                for t in params_per_tp:
                    gatep, upp = t.chunk(2)
                    gate = torch.cat([gate, gatep])
                    up = torch.cat([up, upp])
                output_state_dict[layer_name + ".mlp.gate_proj.weight"] = gate.to(dtype).clone().detach().contiguous()
                output_state_dict[layer_name + ".mlp.up_proj.weight"] = up.to(dtype).clone().detach().contiguous()

            # Transpose the weights.
            elif weight_or_bias == "weight":
                out_name = megatron_to_transformers[op_name]
                output_state_dict[layer_name + out_name + "weight"] = params

            # Copy the bias.
            elif weight_or_bias == "bias":
                out_name = megatron_to_transformers[op_name]
                output_state_dict[layer_name + out_name + "bias"] = params
            
            rotary_base = 10000
            inv_freq = 1.0 / (rotary_base ** (torch.arange(0, hidden_size_per_head * tp_state_dicts[0]['args'].rotary_percent, 2).float() / int(hidden_size_per_head * tp_state_dicts[0]['args'].rotary_percent)))
            output_state_dict[layer_name + '.self_attn.rotary_emb.inv_freq'] = inv_freq

    if config.num_hidden_layers != (layer_idx + 1):
        raise ValueError(f"Expected {config.n_layer} layers but found {layer_idx + 1}")

    # The final layernorm.
    print("Converting final layernorm")
    params = get_element_from_dict_by_path(tp_state_dicts[0], str(path))
    output_state_dict["model.norm.weight"] = params["final_norm.weight"].to(dtype)

    # For LM head, transformers' wants the matrix to weight embeddings.
    print("Converting LM head")
    lm_embeddings = torch.cat(
        [
            get_element_from_dict_by_path(
                tp_state_dicts[tp_rank], "model.language_model.output_layer.weight"
            )
            for tp_rank in range(tp_size)
        ],
        dim=0,
    )
    lm_embeddings = lm_embeddings[:vocab_size].to(dtype).clone().detach().contiguous()
    output_state_dict["lm_head.weight"] = lm_embeddings

    # It should be done!
    print("Conversion from Megatron-LM to Transformers is done!")

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    # Save tokenizer based on args
    copy_tokenizer(args=args)

    # Store the config to file.
    print("Saving config")
    config.save_pretrained(args.save_path)

    # Store the state_dict to file.
    max_shard_size = int(args.max_shard_size) if args.max_shard_size.isdigit() else args.max_shard_size
    shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)

    # Save the model
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(args.save_path, shard_file))

    if index is None:
        print(f"Model weights saved in {os.path.join(args.save_path, WEIGHTS_NAME)}")
    else:
        save_index_file = os.path.join(args.save_path, WEIGHTS_INDEX_NAME)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        print(
            f"The model is bigger than the maximum size per checkpoint ({args.max_shard_size}) and is going to be "
            f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
            f"index located at {save_index_file}."
        )



def main():
    parser = argparse.ArgumentParser()
    parser = add_checkpointing_args(parser)
    parser = add_megatron_checkpoint_args(parser)
    parser = add_transformers_checkpoint_args(parser)
    args = parser.parse_args()

    if args.convert_checkpoint_from_megatron_to_transformers:
        convert_checkpoint_from_megatron_to_transformers(args)
    else:
        convert_checkpoint_from_transformers_to_megatron(args)


if __name__ == "__main__":
    main()
