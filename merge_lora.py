import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

BASE_MODEL_ID = "OpenPipe/Qwen3-14B-Instruct" 

ADAPTER_PATH = "/home/ubuntu/real_tab/.art/data-analyst-real-14b/models/tabmwp-real-14b-1/checkpoints/0231"

OUTPUT_DIR = "./merged_14b_final"

print(f"Loading base model: {BASE_MODEL_ID}...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print(f"Loading LoRA adapter: {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("Merging weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Merge Complete! Now you can serve this model with vLLM.")