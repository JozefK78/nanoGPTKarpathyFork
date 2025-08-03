#!/bin/bash

# This script sets up the environment for data preparation.
# It redirects cache directories and sets up Python environment variables.
# 'set -e' ensures the script will exit immediately if any command fails.
set -e

echo "--- Setting up Environment ---"

# --- PART 1: THE *CORRECT* SYMLINK TRICK ---
echo "[Step 1/2] Preparing the OS-level redirect for the Hugging Face cache..."

# The home directory in most Docker containers is /root
# We are targeting the default cache location that is filling up your 'overlay' filesystem.
HF_DEFAULT_CACHE_DIR_IN_CONTAINER="/root/.cache/huggingface"
NETWORK_DRIVE_BASE="/workspace"
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

# --- PART 2: SETUP PYTHON ENVIRONMENT ---
# Although the symlink is the main fix, setting these is still good practice.
export HF_HOME="${NETWORK_CACHE_TARGET}"
export HF_DATASETS_CACHE="${NETWORK_CACHE_TARGET}/datasets"

echo "--- Environment setup complete. ---"