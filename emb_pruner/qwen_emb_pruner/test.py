
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/nvme/qd/ckpt/pruned_qwen",trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("/nvme/hf_models/qwen/Qwen-1_8B")
text = '''As you know, the surface of Earth is not flat. Some places are high, and some places are low. For example, mountain ranges like the Sierra Nevada in California or the Andes in South America are high above the surrounding areas. An accurate location must take into account the third dimension. Elevation is the height above or below sea level. Sea level refers to the height of the ocean’s surface. This is the midpoint between high and low tide. Sea level can vary from place to place, but scientists base their elevation measurements on the average, or mean, sea level to make sure they have a standard reference point.
Question: What is the height above or below sea level called?
Answer: depth'''

text = '''’'''
res = tokenizer.encode(text)
# import pdb
# pdb.set_trace()

