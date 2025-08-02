import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# --- DYNAMIC AND SAFE PROCESS CONFIGURATION ---
# try:
#     total_cores = os.cpu_count()
#     safe_proc_limit = max(1, int(total_cores * 0.8))
#     print(f"Detected {total_cores} total logical cores. Using a safe process limit of {safe_proc_limit}.")
#     num_proc = safe_proc_limit
# except NotImplementedError:
#     num_proc = 32
#     print(f"Could not determine CPU count. Defaulting to {num_proc} processes.")

num_proc = 32


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
os.environ["HF_DATASETS_IN_PROGRESS_DIR"] = os.path.join(TMP_DIR, "hf_datasets_in_progress")
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
num_proc_load_dataset = num_proc
enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    remote_name = "sample-10BT"
    print(f"Loading dataset with {num_proc_load_dataset} processes...")
    
    dataset = load_dataset("HuggingFaceFW/fineweb-edu",
        name=remote_name,
        split="train",
        num_proc=num_proc_load_dataset,
        cache_dir=os.environ["HF_DATASETS_CACHE"],
    )

    split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # --- FINAL, HIGH-PERFORMANCE WRITING SECTION ---
    for split, dset in tokenized.items():
        output_filename = os.path.join(DATA_ROOT, f'{split}.bin')
        if os.path.exists(output_filename):
            print(f"{output_filename} already exists, skipping")
            continue

        arr_len = np.sum(dset['len'], dtype=np.uint64)
        dtype = np.uint16
        arr = np.memmap(output_filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        # This chunk_size is a tunable parameter. It defines how many documents
        # to accumulate in RAM before writing to the disk.
        # A larger size means more RAM usage but fewer, more efficient writes.
        chunk_size = 100_000 
        
        # A buffer to hold the token IDs for the current chunk
        batch_ids = []
        idx = 0
        
        with tqdm(total=len(dset), desc=f'writing {output_filename}') as pbar:
            for example in dset:
                # Add the new example's IDs to our buffer
                batch_ids.append(example['ids'])
                
                # If the buffer is full, write it to disk
                if len(batch_ids) >= chunk_size:
                    # Concatenate all the lists in the buffer into one big array
                    arr_batch = np.concatenate(batch_ids)
                    # Write this single large array to the memmap file
                    arr[idx : idx + len(arr_batch)] = arr_batch
                    # Update our position in the file
                    idx += len(arr_batch)
                    # Update the progress bar
                    pbar.update(len(batch_ids))
                    # Reset the buffer
                    batch_ids = []

            # After the loop, there might be remaining examples in the buffer
            if batch_ids:
                arr_batch = np.concatenate(batch_ids)
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
                pbar.update(len(batch_ids))
        
        arr.flush()

    print("Data preparation complete. All files written to the network drive.")