import os
import shutil
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, load_from_disk

# --- CONFIGURATION ---
SAVE_TOKENIZED_DATASET = False # If True, saves a cache of the tokenized dataset for faster re-runs.
num_proc = 9 # Using the fixed, safe number of processes.
print(f"--> Using a fixed number of {num_proc} processes.")

# --- PATHS (All on the robust network drive) ---
NETWORK_DRIVE_BASE = "/workspace"
DATA_ROOT = os.path.join(NETWORK_DRIVE_BASE, "nanoGPTKarpathyFork", "data", "fineweb_edu_10BT")
TOKENIZED_DATASET_PATH = os.path.join(DATA_ROOT, "tokenized_dataset_cache")

# The OS-level symlink from run.sh handles the real work.
# We create the final output directory here.
os.makedirs(DATA_ROOT, exist_ok=True)

if __name__ == '__main__':
    # --- STAGE 1: TOKENIZATION (Resumable) ---
    if SAVE_TOKENIZED_DATASET and os.path.exists(TOKENIZED_DATASET_PATH):
        print(f"--> Found cached tokenized dataset at {TOKENIZED_DATASET_PATH}. Loading from disk...")
        tokenized = load_from_disk(TOKENIZED_DATASET_PATH)
    else:
        if SAVE_TOKENIZED_DATASET:
            print("--> No cached tokenized dataset found. Starting full processing pipeline...")
        else:
            print("--> Starting full processing pipeline (tokenized dataset caching is disabled)...")
        # This command will now work because its attempts to write to /root/.cache/huggingface
        # are being redirected to /workspace/hf_cache by the OS.
        print(f"--> Loading raw dataset with {num_proc} processes...")
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", num_proc=num_proc)
        
        split_dataset = dataset['train'].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
        split_dataset['val'] = split_dataset.pop('test')

        enc = tiktoken.get_encoding("gpt2")
        def process(example):
            ids = enc.encode_ordinary(example['text'])
            ids.append(enc.eot_token)
            return {'ids': ids, 'len': len(ids)}

        print(f"--> Tokenizing dataset with {num_proc} processes...")
        tokenized = split_dataset.map(
            process, remove_columns=['text'], desc="Tokenizing the splits", num_proc=num_proc,
        )

        if SAVE_TOKENIZED_DATASET:
            print(f"--> Saving tokenized dataset to {TOKENIZED_DATASET_PATH} for future runs...")
            tokenized.save_to_disk(TOKENIZED_DATASET_PATH)

    # --- STAGE 2: WRITING BINARY FILES ---
    for split, dset in tokenized.items():
        final_output_filename = os.path.join(DATA_ROOT, f'{split}.bin')
        if os.path.exists(final_output_filename):
            print(f"--> Final file {final_output_filename} already exists. Skipping.")
            continue
        
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        dtype = np.uint16
        arr = np.memmap(final_output_filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        idx = 0
        for example in tqdm(dset, desc=f'Writing {split}.bin'):
            arr[idx : idx + len(example['ids'])] = example['ids']
            idx += len(example['ids'])
        
        arr.flush()

    print("--> Data preparation complete.")