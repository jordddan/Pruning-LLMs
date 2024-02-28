import json
import os

workspace = "/nvme/data_evaluation/LEO_mmlu_new_code/data/5shot"
res = []
for root, dirs, files in os.walk(workspace):
    for file in sorted(files):
        path = os.path.join(root, file) 
        with open(path,'r') as f:
            data = json.load(f)
        for line in data:
            res.append(line["data"])    
with open("/home/amax/qd/Megatron-LM/tools/emb_pruner/mmlu.json",'w') as f:
    json.dump(res,f,indent=1)