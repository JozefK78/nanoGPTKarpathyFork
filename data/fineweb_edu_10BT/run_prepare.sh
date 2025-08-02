#!/bin/bash

# This is the definitive launch script. It will solve the storage problem.
# 'set -e' ensures the script will exit immediately if any command fails.
set -e

echo "--- Starting Robust Data Preparation ---"

# --- PART 1: CLEANUP ---
echo "[Step 1/4] Cleaning up previous runs from the network drive..."
NETWORK_DRIVE_BASE="/workspace"
rm -rf "${NETWORK_DRIVE_BASE}/hf_cache"
rm -rf "${NETWORK_DRIVE_BASE}/tmp" # This is now our general-purpose temp dir
rm -rf "${NETWORK_DRIVE_BASE}/nanoGPTKarpathyFork/data/fineweb_edu_10BT/tokenized_dataset_cache"
rm -f  "${NETWORK_DRIVE_BASE}/nanoGPTKarpathyFork/data/fineweb_edu_10BT"/*.bin
echo "Cleanup complete."

# --- PART 2: THE *CORRECT* SYMLINK TRICK ---
echo "[Step 2/4] Preparing the OS-level redirect for the Hugging Face cache..."

# The home directory in most Docker containers is /root
# We are targeting the default cache location that is filling up your 'overlay' filesystem.
HF_DEFAULT_CACHE_DIR_IN_CONTAINER="/root/.cache/huggingface"
NETWORK_CACHE_TARGET="${NETWORK_DRIVE_BASE}/hf_cache"

# First, ensure the parent directory exists inside the container
mkdir -p /root/.cache

# Delete the problematic directory on the local disk if it exists
rm -rf "${HF_DEFAULT_CACHE_DIR_IN_CONTAINER}"

# Create our target directory on the network drive
mkdir -p "${NETWORK_CACHE_TARGET}"

# Create the symbolic link. This is the most critical step.
# Any attempt to write to "/root/.cache/huggingface" will now be redirected to "/workspace/hf_cache".
ln -s "${NETWORK_CACHE_TARGET}" "${HF_DEFAULT_CACHE_DIR_IN_CONTAINER}"

echo "Redirect successful. All Hugging Face cache operations will now use the network drive."

# --- PART 3: SETUP PYTHON ENVIRONMENT ---
# Although the symlink is the main fix, setting these is still good practice.
export HF_HOME="${NETWORK_CACHE_TARGET}"
export HF_DATASETS_CACHE="${NETWORK_CACHE_TARGET}/datasets"

# --- PART 4: EXECUTE THE PYTHON SCRIPT ---
echo "[Step 4/4] Running the Python preparation script..."
python prepare_final.py

echo "--- All steps completed successfully! ---"