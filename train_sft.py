import asyncio
import json
import os
import re
import pandas as pd
from io import StringIO
import sys
from datasets import load_dataset
from openai import AsyncOpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()


LOCAL_DB_PATH = os.path.abspath("./tabmwp_database")

def prepare_sft_data():
    print("Loading TableSenseAI/TabMWP for SFT...")
    ds = load_dataset("TableSenseAI/TabMWP", split="train")
    split_ds = ds.train_test_split(test_size=0.01, seed=42)
    
    train_examples = []
    val_examples = []

    def process_example(row) -> dict | None:
        # Handle context - might be string or dict
        ctx = row["context"]
        if isinstance(ctx, str):
            try:
                ctx = json.loads(ctx)
            except json.JSONDecodeError:
                print(f"Failed to parse context: {ctx}")
                return None
        
        rel_path = ctx.get("csv")
        
        if not rel_path:
            return None
            
        full_path = os.path.join(LOCAL_DB_PATH, rel_path)
        if not os.path.exists(full_path):
            print(f"File not found, skipping: {full_path}")
            return None

        choices_str = f"\nChoices: {json.dumps(row['choices'], ensure_ascii=False)}" if row.get("choices") else ""

        conversation = [
            {
                "from": "system",
                "value": f"""You are an expert data analyst using Python and Pandas.

Rules you MUST follow:
1. ALWAYS start by reading the CSV file with pd.read_csv
2. Use ```python ... ``` blocks to write code
3. Inspect the data (head, columns, dtypes, describe, value_counts etc.)
4. Solve the question using code execution only (NO mental math)
5. When you have the final answer, output it EXACTLY like this:
   Final Answer: [your answer here]

The CSV file path is: {full_path}"""
            },
            {
                "from": "human",
                "value": f"Question: {row['utterance']}{choices_str}"
            },
            {
                "from": "gpt",
                "value": f"""Understood. I'll start by reading and inspecting the table.

```python
import pandas as pd
df = pd.read_csv("{full_path}")
print(df.head(8))
print("\\nColumns:", df.columns.tolist())
print("\\nData types:\\n", df.dtypes)
print("\\nShape:", df.shape)
```"""
            },
            {
                "from": "human",
                "value": "\nObservation:\n"
                           "Columns: ['x', 'y', 'z']\n"
                           "Data types:\n x    int64\n y    object\n z    float64\n"
                           "Shape: (10, 3)\n"
            },
            {
                "from": "gpt",
                "value": f"""From the data inspection and required calculation, the answer is '{row['target_value']}'.

Final Answer: {row['target_value']}"""
            }
        ]
        
        return {"conversations": conversation}

    train_df = split_ds['train'].to_pandas()
    for idx, row in train_df.iterrows():
        ex = process_example(row)
        if ex:
            train_examples.append(ex)
    
    val_df = split_ds['test'].to_pandas()
    for idx, row in val_df.head(50).iterrows():
        ex = process_example(row)
        if ex:
            val_examples.append(ex)

    print(f"SFT data prepared! Train: {len(train_examples)}, Val: {len(val_examples)}")
    
    with open("tabmwp_sft_train.jsonl", "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    with open("tabmwp_sft_val.jsonl", "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    print("Saved to tabmwp_sft_train.jsonl and tabmwp_sft_val.jsonl")

    print("\n" + "="*60)
    print("Example conversation format:")
    print("="*60)
    print(json.dumps(train_examples[0], indent=2, ensure_ascii=False))
    print("="*60 + "\n")
    
    return train_examples, val_examples


def create_llamafactory_config():
    
    os.makedirs("data", exist_ok=True)

    dataset_info = {
        "tabmwp_train": {
            "file_name": "../tabmwp_sft_train.jsonl",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            }
        },
        "tabmwp_val": {
            "file_name": "../tabmwp_sft_val.jsonl",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            }
        }
    }
    
    with open("data/dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print("✓ Created data/dataset_info.json")
    
    training_config = """### Model
model_name_or_path: OpenPipe/Qwen3-14B-Instruct

### Method
stage: sft
do_train: true
finetuning_type: lora
lora_target: q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj

### Dataset
dataset: tabmwp_train
template: qwen
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 16

### Output
output_dir: ./sft_tabmwp_qwen14b
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### Train
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 5.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### Eval
val_size: 0.01
per_device_eval_batch_size: 2
eval_strategy: steps
eval_steps: 500
"""
    
    with open("train_config.yaml", "w", encoding="utf-8") as f:
        f.write(training_config)
    
    print("✓ Created train_config.yaml")
    
    train_script = """#!/bin/bash

llamafactory-cli train train_config.yaml

# llamafactory-cli train \\
#     --model_name_or_path OpenPipe/Qwen3-14B-Instruct \\
#     --stage sft \\
#     --do_train \\
#     --finetuning_type lora \\
#     --lora_target q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj \\
#     --dataset tabmwp_train \\
#     --template qwen \\
#     --cutoff_len 4096 \\
#     --output_dir ./sft_tabmwp_qwen14b \\
#     --per_device_train_batch_size 2 \\
#     --gradient_accumulation_steps 8 \\
#     --learning_rate 5e-5 \\
#     --num_train_epochs 3.0 \\
#     --lr_scheduler_type cosine \\
#     --warmup_ratio 0.1 \\
#     --logging_steps 10 \\
#     --save_steps 500 \\
#     --bf16 \\
#     --val_size 0.01 \\
#     --eval_strategy steps \\
#     --eval_steps 500
"""
    
    with open("train.sh", "w", encoding="utf-8") as f:
        f.write(train_script)
    
    os.chmod("train.sh", 0o755)
    print("✓ Created train.sh")


def print_usage_instructions():
    print("\n" + "="*60)
    print("SFT TRAINING SETUP COMPLETE!")
    print("="*60)
    
    print("\n📁 Files created:")
    print("  - tabmwp_sft_train.jsonl   (training data)")
    print("  - tabmwp_sft_val.jsonl     (validation data)")
    print("  - data/dataset_info.json   (dataset config)")
    print("  - train_config.yaml        (training config)")
    print("  - train.sh                 (training script)")
    
    print("\n🚀 To start training:")
    print("  Method 1 (recommended):")
    print("    ./train.sh")
    print("\n  Method 2:")
    print("    llamafactory-cli train train_config.yaml")
    
    print("\n📊 Monitor training:")
    print("  - Logs: ./sft_tabmwp_qwen14b/trainer_log.jsonl")
    print("  - Loss plot: ./sft_tabmwp_qwen14b/training_loss.png")
    
    print("\n💡 Tips:")
    print("  - Adjust batch_size if you get OOM errors")
    print("  - Default: batch_size=2, gradient_accumulation=8 (effective batch=16)")
    print("  - Training will take several hours on a single GPU")
    
    print("\n🔧 After training, merge LoRA weights:")
    print("  llamafactory-cli export \\")
    print("    --model_name_or_path OpenPipe/Qwen3-14B-Instruct \\")
    print("    --adapter_name_or_path ./sft_tabmwp_qwen14b \\")
    print("    --template qwen \\")
    print("    --finetuning_type lora \\")
    print("    --export_dir ./merged_model \\")
    print("    --export_size 2 \\")
    print("    --export_device cpu")
    print("="*60 + "\n")


if __name__ == "__main__":
    train_data, val_data = prepare_sft_data()
    
    create_llamafactory_config()
    
    print_usage_instructions()