import os
import time
from huggingface_hub import list_repo_files, hf_hub_download
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

REPO_ID = "TableSenseAI/TabMWP"
REPO_TYPE = "dataset"
LOCAL_DIR = "./tabmwp_database"

def download_file(file_path):
    full_local_path = os.path.join(LOCAL_DIR, file_path)
    
    if os.path.exists(full_local_path):
        return "skipped"

    try:
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=file_path,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )
        return "downloaded"
    except Exception as e:
        if "429" in str(e) or "rate limit" in str(e).lower():
            return "ratelimit"
        return f"error: {e}"

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    print("Fetching file list from Hugging Face (costs 1 API call)...")
    try:
        all_files = list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
        
        target_files = [f for f in all_files if f.endswith(('.csv', '.json', '.parquet'))]
        print(f"Found {len(target_files)} target files in repo.")
        
    except Exception as e:
        print(f"Failed to fetch file list: {e}")
        return

    print("Starting smart download (skipping existing files locally)...")

    while True:
        hit_rate_limit = False
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(download_file, f): f for f in target_files}
            progress = tqdm(as_completed(futures), total=len(target_files), desc="Downloading")
            
            for future in progress:
                result = future.result()
                
                if result == "ratelimit":
                    hit_rate_limit = True
                    continue
                elif result == "skipped":
                    pass
        
        if hit_rate_limit:
            print("\n⚠️ Hit Rate Limit! The script will verify files locally and pause.")
            print("Sleeping for 6 minutes before retrying missing files...")
            time.sleep(360)
            print("Resuming...")
        else:
            print("\n✅ All files processed successfully!")
            break

if __name__ == "__main__":
    main()