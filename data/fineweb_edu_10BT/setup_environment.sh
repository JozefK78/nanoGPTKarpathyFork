#!/bin/bash

# This script sets up the environment for data preparation.
# It redirects cache directories and sets up Python environment variables.
# 'set -e' ensures the script will exit immediately if any command fails.
set -e

echo "--- Setting up Environment ---"

# --- PART 1: ENVIRONMENT-AWARE CACHE REDIRECTION ---
echo "[Step 1/2] Checking for RunPod environment for cache redirection..."

# Only perform the symlink trick if we are in a RunPod-like environment
if [ -d "/workspace" ]; then
    echo "RunPod environment detected. Redirecting Hugging Face cache to /workspace."

    # The home directory in most Docker containers is /root
    HF_DEFAULT_CACHE_DIR_IN_CONTAINER="/root/.cache/huggingface"
    NETWORK_DRIVE_BASE="/workspace"
    NETWORK_CACHE_TARGET="${NETWORK_DRIVE_BASE}/hf_cache"

    # First, ensure the parent directory exists inside the container
    mkdir -p /root/.cache

    # Delete the problematic directory on the local disk if it exists
    rm -rf "${HF_DEFAULT_CACHE_DIR_IN_CONTAINER}"

    # Create our target directory on the network drive
    mkdir -p "${NETWORK_CACHE_TARGET}"

    # Create the symbolic link.
    ln -s "${NETWORK_CACHE_TARGET}" "${HF_DEFAULT_CACHE_DIR_IN_CONTAINER}"

    echo "Redirect successful. All Hugging Face cache operations will now use the network drive."

    # Set environment variables to point to the new cache location
    export HF_HOME="${NETWORK_CACHE_TARGET}"
    export HF_DATASETS_CACHE="${NETWORK_CACHE_TARGET}/datasets"
else
    echo "Local environment detected. Skipping cache redirection."
fi

# --- PART 2: Finalizing Setup ---
echo "--- Environment setup complete. ---"