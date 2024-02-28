ckpt_path="/nvme/hf_models/qwen/Qwen-1_8B"
new_path="/nvme/qd/ckpt/pruned_qwen"
rm -rf $new_path
cp -r $ckpt_path $new_path
echo "create new ckpt dir"

python /home/amax/qd/LEO_V2_DEV/LEO_debug/emb_pruner/qwen_emb_pruner/prune_tiktoken_and_emb.py \
    --hf_dir $new_path 
    