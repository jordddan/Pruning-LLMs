# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""GPT-2 model."""

import torch

from megatron import get_args
from megatron.core import tensor_parallel
from .module import MegatronModule

from .enums import AttnMaskType
from .language_model import parallel_lm_logits
from .language_model import get_language_model

from megatron.core.parallel_state import (
    get_global_memory_buffer,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from megatron.model.utils import (
    openai_gelu,
    get_linear_layer,
    init_method_normal,
    scaled_init_method_normal,
    get_random_index,
    get_random_mask,
    get_tensor_per_partition
)

def post_language_model_processing(lm_output, labels, logit_weights,
                                   parallel_output,
                                   fp16_lm_cross_entropy):

    # Output. Format [s b h]
    output = parallel_lm_logits(
        lm_output,
        logit_weights,
        parallel_output)

    if labels is None:
        # [s b h] => [b s h]
        return output.transpose(0,1).contiguous()
    else:
        # [b s] => [s b]
        labels = labels.transpose(0,1).contiguous()
        if fp16_lm_cross_entropy:
            assert output.dtype == torch.half
            loss = tensor_parallel.vocab_parallel_cross_entropy(output, labels)
        else:
            loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), labels)
        
        # [s b] => [b, s]
        loss = loss.transpose(0,1).contiguous()
        return loss


class GPTModel(MegatronModule):
    """GPT-2 Language model."""

    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True):
        args = get_args()
        super().__init__(config=config, share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights)

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights

        self.language_model, self._language_model_key = get_language_model(
            config=config,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            pre_process=self.pre_process,
            post_process=self.post_process)
        
        if not args.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings()
        self.prune_zs = None
        if args.is_prune:
            self.prune_zs = self.init_prune_zs(args)
    
    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, input_ids, position_ids, attention_mask,
                retriever_input_ids=None,
                retriever_position_ids=None,
                retriever_attn_mask=None,
                labels=None, tokentype_ids=None, inference_params=None):
        # torch.save(input_ids,f"/data/megatron_ckpt/llama_test/tensor{torch.distributed.get_rank()}")
        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            retriever_input_ids=retriever_input_ids,
            retriever_position_ids=retriever_position_ids,
            retriever_attn_mask=retriever_attn_mask,
            inference_params=inference_params)

        if self.post_process:
            return post_language_model_processing(
                lm_output, labels,
                self.language_model.output_layer.weight if self.untie_embeddings_and_output_weights else self.shared_embedding_or_output_weight(),
                self.parallel_output,
                self.fp16_lm_cross_entropy)
        else:
            return lm_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        args = get_args()
        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars)
        # Save word_embeddings.
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(prefix=prefix,
                                                  keep_vars=keep_vars)
        
        if self.prune_zs is not None:
            state_dict_["zs"] = self.prune_zs

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Load word_embeddings.
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
        # import rpdb; rpdb.set_trace('0.0.0.0', 1024+torch.distributed.get_rank())
        args = get_args()
        # import pdb
        # pdb.set_trace()
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)


    def init_prune_zs(self, args):
        zs = {}
        hidden_size = args.hidden_size
        hidden_size_remain = args.hidden_size_remain
        num_attention_heads = args.num_attention_heads 
        num_attention_heads_remain = args.num_attention_heads_remain 
        
        ffn_hidden_size = args.ffn_hidden_size
        ffn_hidden_size_remain = args.ffn_hidden_size_remain

        hidden_mask = get_random_mask(hidden_size,hidden_size_remain)
        hidden_mask.requires_grad = False
        zs["hidden_mask"] = hidden_mask

        hidden_index = torch.where(hidden_mask==True)[0]
        hidden_index.requires_grad = False
        zs["hidden_index"] = hidden_index
        head_masks = []
        head_indexes = []
        intermediate_masks = []
        intermediate_indexes = []

        layer_num = args.num_layers

        parallel_size = args.tensor_model_parallel_size
        model_parallel_world_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        assert args.prune_type in ["balance","unbalance"], "no such prune_type"
        for i in range(layer_num):
            if args.prune_type == "unbalance":
                head_mask = get_random_mask(num_attention_heads, num_attention_heads_remain)
                head_mask_per_partition = get_tensor_per_partition(head_mask,rank,model_parallel_world_size)
                head_masks.append(head_mask_per_partition)
                head_indexes.append(torch.where(head_mask_per_partition==True)[0])

                

                intermediate_mask = get_random_mask(ffn_hidden_size,ffn_hidden_size_remain)
                intermediate_mask_per_partition = get_tensor_per_partition(intermediate_mask,rank,model_parallel_world_size)
                intermediate_masks.append(intermediate_mask_per_partition)
                intermediate_indexes.append(torch.where(intermediate_mask_per_partition==True)[0])

            elif args.prune_type == "balance":
                assert num_attention_heads_remain % model_parallel_world_size == 0, "new heads cannot be devided by tp size"
                assert ffn_hidden_size_remain % model_parallel_world_size == 0, "new ffn_hidden_size cannot be devided by tp size"

                # get head zs for each partition 
                num_attention_heads_tp = num_attention_heads // model_parallel_world_size
                num_attention_heads_remain_tp = num_attention_heads_remain // model_parallel_world_size
                
                head_mask_per_partition = get_random_mask(num_attention_heads_tp, num_attention_heads_remain_tp)
                head_masks.append(head_mask_per_partition)
                head_indexes.append(torch.where(head_mask_per_partition==True)[0])

                # get ffn zs for each partition 
                ffn_hidden_size_tp = ffn_hidden_size // model_parallel_world_size
                ffn_hidden_size_remain_tp = ffn_hidden_size_remain // model_parallel_world_size

                intermediate_mask_per_partition = get_random_mask(ffn_hidden_size_tp, ffn_hidden_size_remain_tp)
                intermediate_masks.append(intermediate_mask_per_partition)
                intermediate_indexes.append(torch.where(intermediate_mask_per_partition==True)[0])


        zs["head_masks"] = head_masks
        zs["intermediate_masks"] = intermediate_masks

        zs["head_indexes"] = head_indexes 
        zs["intermediate_indexes"] = intermediate_indexes
        # save_path = os.path.join(args.save,f"rank{rank}.pt")
        # torch.save(zs,save_path)
        # import rpdb; rpdb.set_trace('0.0.0.0', 1024+torch.distributed.get_rank())
        return zs
    
    def prune(self, args):

        self.language_model.prune(self.prune_zs)


