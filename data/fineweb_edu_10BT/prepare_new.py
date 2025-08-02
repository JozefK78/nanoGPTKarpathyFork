import os
from tqdm import tqdm
import numpy as np
import tiktoken
# We do NOT import or use DownloadConfig
from datasets import load_dataset

# --- Centralized Path Configuration ---
NETWORK_DRIVE_BASE = "/workspace"
DATA_ROOT = os.path.join(NETWORK_DRIVE_BASE, "nanoGPTKarpathyFork", "data", "fineweb_edu_10BT")
HF_CACHE_DIR = os.path.join(NETWORK_DRIVE_BASE, "hf_cache")
TMP_DIR = os.path.join(NETWORK_DRIVE_BASE, "tmp")

# --- Environment Variable Setup ---
# This is the correct and primary way to control caches and temp files in datasets v4.0.0
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "transformers")
os.environ["HF_MODULES_CACHE"] = os.path.join(HF_CACHE_DIR, "modules")
# This is the key variable that the dataset builder will use for its temporary files.
os.environ["HF_DATASETS_IN_PROGRESS_DIR"] = os.path.join(TMP_DIR, "hf_datasets_in_progress")

# Set temporary directory for all other libraries
os.environ["TMPDIR"] = TMP_DIR
os.environ["TEMP"] = TMP_DIR
os.environ["TMP"] = TMP_DIR

# --- Create all necessary directories ---
# This ensures the library doesn't fail trying to create a directory.
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_MODULES_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_IN_PROGRESS_DIR"], exist_ok=True)
os.makedirs(os.environ["TMPDIR"], exist_ok=True)
os.makedirs(DATA_ROOT, exist_ok=True)

# --- Processing Configuration ---
num_proc = 8
num_proc_load_dataset = num_proc
enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # remote_name for HuggingFaceFW/fineweb-edu
    remote_name = "sample-10BT"
    print("Loading dataset using environment variables for temporary file placement...")
    
    # This call correctly uses the environment variables set above for caching and temp files.
    dataset = load_dataset("HuggingFaceFW/fineweb-edu",
        name=remote_name,
        split="train",
        num_proc=num_proc_load_dataset,
        cache_dir=os.environ["HF_DATASETS_CACHE"],
    )

    # owt by default only contains the 'train' split, so create a test split
    split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # --- Optimized and Corrected Writing Section ---
    for split, dset in tokenized.items():
        output_filename = os.path.join(DATA_ROOT, f'{split}.bin')
        if os.path.exists(output_filename):
            print(f"{output_filename} already exists, skipping")
            continue

        arr_len = np.sum(dset['len'], dtype=np.uint64)
        dtype = np.uint16
        arr = np.memmap(output_filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        idx = 0
        for example in tqdm(dset, desc=f'writing {output_filename}'):
            ids = example['ids']
            arr[idx : idx + len(ids)] = ids
            idx += len(ids)
        
        arr.flush()

    print("Data preparation complete. All files written to the network drive.")