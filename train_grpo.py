import asyncio
import json
import os
import re
import pandas as pd
from io import StringIO
import sys
import concurrent.futures
from dotenv import load_dotenv
from datasets import load_dataset
import art
from art.local import LocalBackend

load_dotenv()

LOCAL_DB_PATH = os.path.abspath("./tabmwp_database")

def prepare_data():
    print("Loading TableSenseAI/TabMWP metadata...")
    try:
        ds = load_dataset("TableSenseAI/TabMWP", split="train")
        split_ds = ds.train_test_split(test_size=0.01, seed=42)
        
        train_data = []
        val_data = []

        def process_split(dataset):
            processed = []
            for row in dataset:
                try:
                    ctx = row["context"]
                    if isinstance(ctx, str):
                        import ast
                        try: ctx = json.loads(ctx)
                        except: ctx = ast.literal_eval(ctx)
                    
                    rel_path = ctx.get("csv")
                    if not rel_path: continue
                    
                    full_path = os.path.join(LOCAL_DB_PATH, rel_path)

                    if not os.path.exists(full_path):
                        continue

                    processed.append({
                        "id": row["id"],
                        "question": row["utterance"],  
                        "answer": row["target_value"],      
                        "file_path": full_path,             
                        "choices": row.get("choices", None) 
                    })
                except Exception as e:
                    continue
            return processed

        train_data = process_split(split_ds['train'])
        val_data = process_split(split_ds['test'])[:20] 
        
        print(f"Data ready! Train: {len(train_data)}, Val: {len(val_data)}")
        return train_data, val_data
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

train_puzzles, val_puzzles = prepare_data()

def execute_pandas_code(code: str) -> str:
    import os
    local_scope = {"pd": pd, "os": os} 

    lines = code.strip().split('\n')
    if not any(line.startswith('print(') for line in lines):
        if '=' not in lines[-1].strip():
            lines[-1] = f"print({lines[-1]})"
    code_to_run = "\n".join(lines)

    def run_code():
        old_stdout = sys.stdout
        redirected = sys.stdout = StringIO()
        try:
            pd.set_option('display.max_rows', 10)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            
            exec(code_to_run, {}, local_scope)
            return redirected.getvalue().strip()
        finally:
            sys.stdout = old_stdout

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_code)
        try:
            output = future.result(timeout=5.0) 
        except concurrent.futures.TimeoutError:
            return "[Error] Execution timed out (5s)."
        except Exception as e:
            return f"[Error] {str(e)}"
    return output if output else "[No Output]"


async def rollout(model: art.Model, puzzle: dict) -> art.Trajectory:
    file_path = puzzle["file_path"]
    
    system_prompt = f"""You are an Autonomous Data Analyst.
    
    Context:
    A CSV file is located at: '{file_path}'
    
    Task:
    Read the file, inspect the data, and answer the user's question.
    
    Response Format:
    1. Use Python to inspect and calculate.
    2. Use Pandas package to read the file.
    3. Once you have the result, output it strictly in this format:
       Final Answer: [The exact answer]
    
    Constraint:
    - NO mental math. You must rely on code execution.
    - Do NOT output 'Final Answer' until you are 100% sure.
    """
    
    choices_str = f"\nChoices: {puzzle['choices']}" if puzzle['choices'] else ""
    user_content = f"Question: {puzzle['question']}{choices_str}"

    messages: art.Messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    acts = [*messages]

    client = model.openai_client()
    max_turns = 6
    final_answer_content = "" 
    tool_used_count = 0
    read_csv_used = False
    
    try:
        for _ in range(max_turns):
            chat_completion = await client.chat.completions.create(
                messages=messages, model=model.name, temperature=0.6,
                stop=["Observation:"], logprobs=True
            )
            choice = chat_completion.choices[0]
            clean_content = (choice.message.content or "").strip()
            choice.message.content = clean_content
            
            messages.append(choice.message)
            acts.append(choice)

            code_matches = re.findall(r"```python(.*?)```", clean_content, re.DOTALL)
            if code_matches:
                tool_used_count += 1
                last_code = code_matches[-1]

                if "read_csv" in last_code:
                    read_csv_used = True
                
                execution_output = execute_pandas_code(last_code)
                if len(execution_output) > 3000:
                    execution_output = execution_output[:3000] + "... [Truncated]"

                obs_msg = {"role": "user", "content": f"\nObservation:\n{execution_output}\n"}
                messages.append(obs_msg)
                acts.append(obs_msg)
            
            elif "Final Answer:" in clean_content:
                parts = clean_content.split("Final Answer:")
                final_answer_content = parts[-1].strip()
                break 
            
            else:
                pass

    except Exception as e:
        print(f"Rollout Error: {e}")
        return art.Trajectory(messages_and_choices=acts, reward=-1.0, metrics={"acc": 0.0})

    if acts and isinstance(acts[-1], dict) and acts[-1].get("role") == "user":
        acts.pop()

    
    ground_truth = str(puzzle["answer"]).lower().strip()
    pred = final_answer_content.lower().strip()
    
    is_correct = False

    if not pred:
        is_correct = False
    else:
        try:
            pred_nums = re.findall(r"[-+]?\d*\.\d+|\d+", pred)
            gt_nums = re.findall(r"[-+]?\d*\.\d+|\d+", ground_truth)
            
            if pred_nums and gt_nums:
                pred_val = float(pred_nums[-1])
                gt_val = float(gt_nums[0])
                if abs(pred_val - gt_val) < 1e-4:
                    is_correct = True
        except: 
            pass

        if not is_correct:
            if ground_truth in pred:
                 is_correct = True

    reward = 0.0

    if not read_csv_used:
        reward = -1.0 
    elif tool_used_count == 0:
        reward = -1.0
    else:
        if is_correct:
            reward += 3.0
            
            if 2 <= tool_used_count <= 3:
                reward += 1.0

        if read_csv_used: reward += 0.2
        
        if final_answer_content: reward += 0.5
        
        # Valid Code Reward
        has_valid = False
        for m in messages:
             c = m.get("content","") if isinstance(m, dict) else m.content
             if "Observation:" in (c or "") and "[Error]" not in (c or ""):
                 has_valid = True
        if has_valid: reward += 0.1

    # Length Penalty
    total_len = sum(len(m.content or "") for m in messages if getattr(m, 'role', '') == 'assistant')
    if total_len > 4000:
        reward -= 0.05 * ((total_len - 4000) / 1000)

    return art.Trajectory(
        messages_and_choices=acts, reward=reward,
        metrics={"acc": 1.0 if is_correct else 0.0, "tools": tool_used_count}
    )

async def main():
    model = art.TrainableModel(
        name="tabmwp-real-14b-1", 
        project="data-analyst-real-14b",
        base_model="OpenPipe/Qwen3-14B-Instruct", 
        _internal_config={
            "init_args": {
                "gpu_memory_utilization": 0.6,
                "max_model_len": 4096, 
                "tensor_parallel_size": 1
            }
        },
    )
    backend = LocalBackend()
    await model.register(backend)
    step = await model.get_step()

    batch_size = 16 

    for i in range(step, 1000):
        print(f"--- Iteration {i} ---")
        
        val_task = art.gather_trajectory_groups(
            (art.TrajectoryGroup(rollout(model, p) for _ in range(1)) for p in val_puzzles),
            pbar_desc="val"
        )

        train_task = art.gather_trajectory_groups(
                (art.TrajectoryGroup(rollout(model, p) for _ in range(2)) for p in train_puzzles[i*batch_size:(i+1)*batch_size]),
                pbar_desc="train"
        )
        
        if val_task:
            val_groups, train_groups = await asyncio.gather(val_task, train_task)
            await model.log(val_groups)
        else:
            train_groups = await train_task

        await model.train(
            train_groups,
            config=art.TrainConfig(
                learning_rate=1e-5,                
                mini_batch_size=1,
                gradient_accumulation_steps=16, 
                kl_coef=0.05
            ),
        )
        
        await model.delete_checkpoints()

if __name__ == "__main__":
    asyncio.run(main())