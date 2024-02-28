ckpt_path="/nvme/hf_models/Llama-2-7b-chat-hf"
new_path="/nvme/qd/ckpt/prune_llama2_chat"
rm -rf $new_path
cp -r $ckpt_path $new_path
echo "create new ckpt dir"
remain_vocab_path="/home/amax/qd/LEO_V2_DEV/LEO_debug/emb_pruner/remain_vocab_ids.pt"

python /home/amax/qd/LEO_V2_DEV/LEO_debug/emb_pruner/prune_spm_and_emb.py \
    --remain_vocab_path $remain_vocab_path \
    --hf_dir $new_path \
    --remain_ids_with_spt /home/amax/qd/LEO_V2_DEV/LEO_debug/emb_pruner/remain_vocab_with_spt.pt
    