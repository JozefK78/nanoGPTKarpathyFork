#!/bin/bash

# This script cleans up all artifacts from the data preparation process.
# It is designed to be run before starting a fresh run of prepare.py.

# --- Configuration ---
NETWORK_DRIVE_BASE="/workspace"

# --- Paths to Clean ---
HF_CACHE_DIR="${NETWORK_DRIVE_BASE}/hf_cache"
TMP_DIR="${NETWORK_DRIVE_BASE}/tmp"
DATA_OUTPUT_DIR="${NETWORK_DRIVE_BASE}/nanoGPTKarpathyFork/data/fineweb_edu_10BT"
TOKENIZED_CACHE_PATH="${DATA_OUTPUT_DIR}/tokenized_dataset_cache"

# Local ephemeral drive temp directories
EPHEMERAL_TMP="/tmp"
#EPHEMERAL_HF_CACHE_DEFAULT=~/.cache/huggingface

echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "!!!                     CLEANUP SCRIPT                     !!!"
echo "!!! This will PERMANENTLY DELETE generated files & caches. !!!"
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo

echo "The following will be DELETED from the NETWORK DRIVE (/workspace):"
echo "  - Full Hugging Face cache: ${HF_CACHE_DIR}"
echo "  - General temp directory:  ${TMP_DIR}"
echo "  - Saved tokenized cache:   ${TOKENIZED_CACHE_PATH}"
echo "  - Final output files:      ${DATA_OUTPUT_DIR}/*.bin"
echo
echo "The following will be DELETED from the EPHEMERAL DRIVE:"
echo "  - Contents of ${EPHEMERAL_TMP}"
echo "  - Default HF cache: ${EPHEMERAL_HF_CACHE_DEFAULT}"
echo

read -p "Are you absolutely sure you want to proceed? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 1
fi

echo "Proceeding with cleanup..."

# Clean Network Drive
echo "Cleaning network drive..."
rm -rf "${HF_CACHE_DIR}"
rm -rf "${TMP_DIR}"
rm -rf "${TOKENIZED_CACHE_PATH}"
rm -f "${DATA_OUTPUT_DIR}"/*.bin

# Clean Ephemeral Drive
echo "Cleaning ephemeral drive..."
find "${EPHEMERAL_TMP}" -mindepth 1 -delete
#rm -rf "${EPHEMERAL_HF_CACHE_DEFAULT}"

echo "Cleanup complete."
