"""Sample Generate GPT"""
import json
import os
import sys
import torch
from megatron import get_args
from megatron import get_tokenizer
from megatron import print_rank_0
from megatron.checkpointing import load_checkpoint
from megatron.core import mpu
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.text_generation import generate_and_post_process
from megatron.arguments import core_transformer_config_from_args

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    config = core_transformer_config_from_args(get_args())
    print_rank_0('building GPT model ...')
    model = GPTModel(config, num_tokentypes=0, parallel_output=False,
                     pre_process=pre_process, post_process=post_process)

    return model

    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True,
                                       'seq_length': 2048})


