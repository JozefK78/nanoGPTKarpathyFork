import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, DownloadConfig # <-- IMPORT DownloadConfig

# --- Centralized Path Configuration ---
NETWORK_DRIVE_BASE = "/workspace"
DATA_ROOT = os.path.join(NETWORK_DRIVE_BASE, "nanoGPTKarpathyFork", "data", "fineweb_edu_10BT")
HF_CACHE_DIR = os.path.join(NETWORK_DRIVE_BASE, "hf_cache")
TMP_DIR = os.path.join(NETWORK_DRIVE_BASE, "tmp")

# --- Environment Variable Setup ---
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "transformers")
os.environ["HF_MODULES_CACHE"] = os.path.join(HF_CACHE_DIR, "modules")
os.environ["HF_DATASETS_IN_PROGRESS_DIR"] = os.path.join(HF_CACHE_DIR, "datasets_in_progress")

# Set temporary directory for all other libraries
os.environ["TMPDIR"] = TMP_DIR
os.environ["TEMP"] = TMP_DIR
os.environ["TMP"] = TMP_DIR

# --- Create all necessary directories ---
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
    # --- The definitive fix for temporary files ---
    # Force the downloader and dataset builder to use your network temp directory.
    download_config = DownloadConfig(
        cache_dir=os.environ["HF_DATASETS_CACHE"],
        temp_dir=os.environ["TMPDIR"]
    )

    # remote_name for HuggingFaceFW/fineweb-edu
    remote_name = "sample-10BT"
    print("Loading dataset. This may take a while, but all temporary files will be on the network drive.")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu",
        name=remote_name,
        split="train",
        num_proc=num_proc_load_dataset,
        download_config=download_config, # <-- PASS THE CONFIG HERE
    )

    # owt by default only contains the 'train' split, so create a test split
    split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

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

    print("Data preparation complete. All files have been written to the network drive.")