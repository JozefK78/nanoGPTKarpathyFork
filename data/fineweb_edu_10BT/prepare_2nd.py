import os
import shutil
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, load_from_disk

# --- CONFIGURATION ---
# The number of processes is now the only main configuration here.
try:
    total_cores = os.cpu_count()
    # Using 75-80% of cores is a safe bet on powerful cloud machines.
    safe_proc_limit = max(1, int(total_cores * 0.75))
    print(f"--> Detected {total_cores} total logical cores. Using a safe process limit of {safe_proc_limit}.")
    num_proc = safe_proc_limit
except NotImplementedError:
    num_proc = 8
    print(f"--> Could not determine CPU count. Defaulting to {num_proc} processes.")

# --- PATHS ---
# All paths are on the robust network drive.
NETWORK_DRIVE_BASE = "/workspace"
DATA_ROOT = os.path.join(NETWORK_DRIVE_BASE, "nanoGPTKarpathyFork", "data", "fineweb_edu_10BT")
HF_CACHE_DIR = os.path.join(NETWORK_DRIVE_BASE, "hf_cache")
TOKENIZED_DATASET_PATH = os.path.join(DATA_ROOT, "tokenized_dataset_cache")

# The OS-level symlink handles all other temp directories. We only need to tell
# Hugging Face where to put its final, permanent cache.
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
os.makedirs(DATA_ROOT, exist_ok=True)

if __name__ == '__main__':
    # --- STAGE 1: TOKENIZATION (Resumable) ---
    if os.path.exists(TOKENIZED_DATASET_PATH):
        print(f"--> Found cached tokenized dataset at {TOKENIZED_DATASET_PATH}. Loading from disk...")
        tokenized = load_from_disk(TOKENIZED_DATASET_PATH)
        print("--> Tokenized dataset loaded successfully.")
    else:
        print("--> No cached tokenized dataset found. Starting full processing pipeline...")
        # Now, when load_dataset tries to write to /tmp, the OS will redirect it to /workspace/tmp.
        # This will solve the "No space left on device" error.
        print(f"--> Loading raw dataset with {num_proc} processes...")
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", num_proc=num_proc)
        
        split_dataset = dataset['train'].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
        split_dataset['val'] = split_dataset.pop('test')

        enc = tiktoken.get_encoding("gpt2")
        def process(example):
            ids = enc.encode_ordinary(example['text'])
            ids.append(enc.eot_token)
            return {'ids': ids, 'len': len(ids)}

        print(f"--> Tokenizing dataset with {num_proc} processes (This is a one-time operation)...")
        tokenized = split_dataset.map(
            process, remove_columns=['text'], desc="Tokenizing the splits", num_proc=num_proc,
        )

        print(f"--> Saving tokenized dataset to {TOKENIZED_DATASET_PATH} for future runs...")
        tokenized.save_to_disk(TOKENIZED_DATASET_PATH)
        print("--> Tokenized dataset checkpoint saved.")

    # --- STAGE 2: WRITING BINARY FILES ---
    # With the temp directory issue solved, we can write directly to the final destination.
    # This avoids the complexity of writing locally and then moving.
    for split, dset in tokenized.items():
        final_output_filename = os.path.join(DATA_ROOT, f'{split}.bin')
        if os.path.exists(final_output_filename):
            print(f"--> Final file {final_output_filename} already exists. Skipping.")
            continue
        
        print(f"--> Writing '{split}' split directly to final destination: {final_output_filename}")
        
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        dtype = np.uint16
        arr = np.memmap(final_output_filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        # This simple streaming write is the cleanest method.
        idx = 0
        for example in tqdm(dset, desc=f'Writing {split}.bin'):
            arr[idx : idx + len(example['ids'])] = example['ids']
            idx += len(example['ids'])
        
        arr.flush()
        print(f"--> Finished writing {split}.bin.")

    print("--> Data preparation complete.")