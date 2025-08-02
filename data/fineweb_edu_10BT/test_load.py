import os
from datasets import load_dataset

# We are only testing the load_dataset command, which is the source of the error.

# --- CONFIGURATION ---
num_proc = 32 # Using the fixed number you requested.
print(f"--> Starting test with {num_proc} processes.")

# --- PATHS ---
NETWORK_DRIVE_BASE = "/workspace"
HF_CACHE_DIR = os.path.join(NETWORK_DRIVE_BASE, "hf_cache")

# --- ENVIRONMENT ---
# Set the final cache location, though we know this is being ignored for temp files.
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

# --- THE TEST ---
# This is the line that fills up the disk. We will watch what happens while it runs.
print("--> Executing load_dataset. Watch the other terminal for disk usage changes.")
try:
    load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", num_proc=num_proc)
except Exception as e:
    print(f"\n--- SCRIPT FAILED AS EXPECTED ---")
    print(e)
    print("--- Please check the output of the 'watch' command in Terminal 2 ---")