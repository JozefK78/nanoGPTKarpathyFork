
## Optimizing Large-Scale Dataset Preparation Scripts

### 1. Introduction

The original `prepare.py` script is excellent for local machines with large, fast SSDs. However, when run in production or cloud environments (like Docker containers, Kubernetes pods, etc.), it suffers from three critical issues:

1.  **Storage Instability:** It writes all temporary files, caches, and final outputs to the script's local directory. In environments with small ephemeral filesystems (e.g., a 30 GB container disk), this will cause a `No space left on device` error and crash the script.
2.  **Poor Performance:** The final writing loop performs millions of tiny, individual write operations. On network-attached storage, which has high latency, this I/O pattern is extremely inefficient and can take hours.
3.  **Lack of Resumability:** The script is not designed to be restarted. If it fails halfway through the multi-hour tokenization process, all progress is lost, and it must start from scratch.

This guide provides step-by-step instructions to refactor the original script into a robust, performant, and resumable version suitable for any environment.

### 2. The Solution Strategy

We will implement a three-part strategy to solve these problems:

1.  **Centralized Path Management:** All file paths will be explicitly defined at the top of the script to use a designated large network drive, preventing any writes to the ephemeral disk.
2.  **Resumable Checkpoints:** We will save the result of the slow tokenization step. On subsequent runs, the script will detect this checkpoint and load it, skipping the most time-consuming part.
3.  **The "Local First" Write Strategy:** To achieve maximum performance, we will write the final `.bin` file to the *fast* local ephemeral drive first and then move the single, completed file to the network drive in one efficient bulk transfer.

---

### 3. Step-by-Step Refactoring Instructions

#### **Step 3.1: Add Necessary Imports and Configure Paths**

At the top of the script, add new imports and define all storage paths in one place. This makes the script configurable and prevents accidental writes to the wrong location.

**Instruction:** Add the following code block to the top of your `prepare.py` script, replacing the original `import` statements.

```python
import os
import shutil
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, load_from_disk

# --- CONFIGURATION ---
# Use a fixed, safe number of processes
num_proc = 32

# --- PATH CONFIGURATION ---
# Define a base path on the large, persistent network drive
NETWORK_DRIVE_BASE = "/workspace" # Or any other network mount point
# Define the final output directory for the .bin files
DATA_ROOT = os.path.join(NETWORK_DRIVE_BASE, "data", "openwebtext")
# Define where to store the Hugging Face cache on the network drive
HF_CACHE_DIR = os.path.join(NETWORK_DRIVE_BASE, "hf_cache")
# Define where to save our resumable tokenization checkpoint
TOKENIZED_DATASET_PATH = os.path.join(DATA_ROOT, "tokenized_dataset_cache")
# Define a temporary directory on the FAST ephemeral disk for the final write
LOCAL_TEMP_BIN_DIR = "/tmp/bin_files"
```

#### **Step 3.2: Make the Script Resumable**

We will wrap the main processing logic in an `if/else` block. The `if` condition will check if the tokenization checkpoint exists. If it does, we skip hours of work. If not, we run the process and save the checkpoint at the end.

**Instruction:** Replace the entire `if __name__ == '__main__':` block from the original script with the following structure.

```python
if __name__ == '__main__':
    # --- STAGE 1: TOKENIZATION (Resumable) ---
    if os.path.exists(TOKENIZED_DATASET_PATH):
        # If the checkpoint exists, load it from the network drive.
        print(f"Found cached tokenized dataset at {TOKENIZED_DATASET_PATH}. Loading...")
        tokenized = load_from_disk(TOKENIZED_DATASET_PATH)
        print("Dataset loaded successfully from checkpoint.")
    else:
        # If no checkpoint, run the full one-time processing pipeline.
        print("No cached dataset found. Starting full processing pipeline...")
        
        # Ensure parent directories exist
        os.makedirs(DATA_ROOT, exist_ok=True)
        
        # Original logic to load and split the dataset.
        # Note the 'cache_dir' argument to force cache location.
        dataset = load_dataset("openwebtext", num_proc=num_proc, cache_dir=HF_CACHE_DIR)
        split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
        split_dataset['val'] = split_dataset.pop('test')

        # Original tokenization function
        enc = tiktoken.get_encoding("gpt2")
        def process(example):
            ids = enc.encode_ordinary(example['text'])
            ids.append(enc.eot_token)
            return {'ids': ids, 'len': len(ids)}

        # Original tokenization map function
        print(f"Tokenizing dataset with {num_proc} processes...")
        tokenized = split_dataset.map(
            process,
            remove_columns=['text'],
            desc="Tokenizing the splits",
            num_proc=num_proc,
        )

        # Save the result as a checkpoint for future runs.
        print(f"Saving tokenized dataset checkpoint to {TOKENIZED_DATASET_PATH}...")
        tokenized.save_to_disk(TOKENIZED_DATASET_PATH)
        print("Checkpoint saved.")
    
    # ... The writing loop will go here next ...
```

#### **Step 3.3: Implement the High-Performance "Local First" Write Loop**

The final step is to replace the slow, original writing loop with our optimized version. This new loop writes to the fast local disk first, then performs a high-speed move to the final network destination.

**Instruction:** Add this code block inside the `if __name__ == '__main__':` block, immediately after the code from the previous step.

```python
    # --- STAGE 2: WRITING BINARY FILES (Optimized "Local First" Strategy) ---
    
    # Ensure the local temporary directory exists and is empty
    os.makedirs(LOCAL_TEMP_BIN_DIR, exist_ok=True)

    for split, dset in tokenized.items():
        final_output_filename = os.path.join(DATA_ROOT, f'{split}.bin')
        
        # Skip if the final file already exists
        if os.path.exists(final_output_filename):
            print(f"Final file {final_output_filename} already exists. Skipping.")
            continue

        # Define the temporary file path on the FAST local disk
        local_temp_filename = os.path.join(LOCAL_TEMP_BIN_DIR, f'{split}.bin')
        
        print(f"Writing '{split}' split to fast local temp file: {local_temp_filename}")
        
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        dtype = np.uint16
        
        # Create the memmap file on the fast local disk
        arr = np.memmap(local_temp_filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        # This simple "streaming" write is extremely fast on a local SSD
        idx = 0
        for example in tqdm(dset, desc=f'Writing {split}.bin locally'):
            arr[idx : idx + len(example['ids'])] = example['ids']
            idx += len(example['ids'])
        
        # Ensure all data is written to the local file before moving
        arr.flush()

        # Perform the final, fast bulk transfer from the local disk to the network drive
        print(f"Moving completed file to final destination: {final_output_filename}")
        shutil.move(local_temp_filename, final_output_filename)
        print(f"Successfully moved {split}.bin to network storage.")

    print("Data preparation complete.")
```

---

### 4. The Final, Robust Script

For clarity, here is the complete, final `prepare.py` script incorporating all the changes.

```python
import os
import shutil
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, load_from_disk

# --- CONFIGURATION ---
num_proc = 32

# --- PATH CONFIGURATION ---
NETWORK_DRIVE_BASE = "/workspace"
DATA_ROOT = os.path.join(NETWORK_DRIVE_BASE, "data", "openwebtext")
HF_CACHE_DIR = os.path.join(NETWORK_DRIVE_BASE, "hf_cache")
TOKENIZED_DATASET_PATH = os.path.join(DATA_ROOT, "tokenized_dataset_cache")
LOCAL_TEMP_BIN_DIR = "/tmp/bin_files"

if __name__ == '__main__':
    # --- STAGE 1: TOKENIZATION (Resumable) ---
    if os.path.exists(TOKENIZED_DATASET_PATH):
        print(f"Found cached tokenized dataset at {TOKENIZED_DATASET_PATH}. Loading...")
        tokenized = load_from_disk(TOKENIZED_DATASET_PATH)
        print("Dataset loaded successfully from checkpoint.")
    else:
        print("No cached dataset found. Starting full processing pipeline...")
        os.makedirs(DATA_ROOT, exist_ok=True)
        
        dataset = load_dataset("openwebtext", num_proc=num_proc, cache_dir=HF_CACHE_DIR)
        split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
        split_dataset['val'] = split_dataset.pop('test')

        enc = tiktoken.get_encoding("gpt2")
        def process(example):
            ids = enc.encode_ordinary(example['text'])
            ids.append(enc.eot_token)
            return {'ids': ids, 'len': len(ids)}

        print(f"Tokenizing dataset with {num_proc} processes...")
        tokenized = split_dataset.map(
            process,
            remove_columns=['text'],
            desc="Tokenizing the splits",
            num_proc=num_proc,
        )

        print(f"Saving tokenized dataset checkpoint to {TOKENIZED_DATASET_PATH}...")
        tokenized.save_to_disk(TOKENIZED_DATASET_PATH)
        print("Checkpoint saved.")

    # --- STAGE 2: WRITING BINARY FILES (Optimized "Local First" Strategy) ---
    os.makedirs(LOCAL_TEMP_BIN_DIR, exist_ok=True)

    for split, dset in tokenized.items():
        final_output_filename = os.path.join(DATA_ROOT, f'{split}.bin')
        
        if os.path.exists(final_output_filename):
            print(f"Final file {final_output_filename} already exists. Skipping.")
            continue

        local_temp_filename = os.path.join(LOCAL_TEMP_BIN_DIR, f'{split}.bin')
        print(f"Writing '{split}' split to fast local temp file: {local_temp_filename}")
        
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        dtype = np.uint16
        arr = np.memmap(local_temp_filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        idx = 0
        for example in tqdm(dset, desc=f'Writing {split}.bin locally'):
            arr[idx : idx + len(example['ids'])] = example['ids']
            idx += len(example['ids'])
        
        arr.flush()

        print(f"Moving completed file to final destination: {final_output_filename}")
        shutil.move(local_temp_filename, final_output_filename)
        print(f"Successfully moved {split}.bin to network storage.")

    print("Data preparation complete.")
```