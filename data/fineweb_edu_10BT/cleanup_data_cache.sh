#!/bin/bash

# --- Configuration ---
# Please verify this is the correct base path for your network drive.
NETWORK_DRIVE_BASE="/workspace"

# --- Define Directories to Clean ---

# 1. Network Drive Directories
# These are the cache, temp, and output directories located on your network drive.
HF_CACHE_DIR="${NETWORK_DRIVE_BASE}/hf_cache"
TMP_DIR="${NETWORK_DRIVE_BASE}/tmp"
DATA_OUTPUT_DIR="${NETWORK_DRIVE_BASE}/nanoGPTKarpathyFork/data/fineweb_edu_10BT"

# 2. Ephemeral Drive Directories (Common Locations)
# These are the default locations that libraries often fall back to if environment
# variables are not set correctly or are ignored by a subprocess.
EPHEMERAL_TMP="/tmp"
# The default Hugging Face cache location is inside the user's home directory.
# The `~` character is a shortcut for the home directory (e.g., /root or /home/user).
EPHEMERAL_HF_CACHE_DEFAULT_1=~/.cache/huggingface
EPHEMERAL_HF_CACHE_DEFAULT_2=/root/.cache/huggingface

# --- Safety Check and Confirmation ---

echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "!!!                     DANGER ZONE                        !!!"
echo "!!! This script will PERMANENTLY DELETE generated files    !!!"
echo "!!! and caches from previous runs.                       !!!"
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo
echo "The following directories on the NETWORK DRIVE will be completely deleted:"
echo "  - Cache: ${HF_CACHE_DIR}"
echo "  - Temp:  ${TMP_DIR}"
echo
echo "The following GENERATED FILES on the NETWORK DRIVE will be deleted:"
echo "  - Output Files: ${DATA_OUTPUT_DIR}/*.bin"
echo "  (Your scripts in this directory will NOT be deleted)"
echo
echo "The following common directories on the EPHEMERAL DRIVE will be cleaned:"
echo "  - Contents of ${EPHEMERAL_TMP}"
echo "  - ${EPHEMERAL_HF_CACHE_DEFAULT_1}"
echo "  - ${EPHEMERAL_HF_CACHE_DEFAULT_2}"
echo
echo "Please double-check these actions. There is no undo."
echo

read -p "Are you absolutely sure you want to proceed? (y/n) " -n 1 -r
echo    # move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cleanup cancelled."
    exit 1
fi

# --- Execution ---

echo
echo "Proceeding with cleanup..."

# Clean Network Drive Caches and Temp
echo "Cleaning network drive cache and temp directories..."
rm -rf "${HF_CACHE_DIR}"
rm -rf "${TMP_DIR}"
echo "Cache and temp directories cleaned."
echo

# Clean Network Drive Output Files (*.bin)
echo "DISABLED! - Deleting *.bin files from the output directory..."
# SAFER: Only deletes files ending in .bin inside the directory.
# The -f flag prevents errors if no *.bin files exist.
#rm -f "${DATA_OUTPUT_DIR}"/*.bin
echo "Generated .bin files deleted."
echo

# Clean Ephemeral Drive
echo "Cleaning ephemeral drive directories..."
# For /tmp, we delete the contents, not the directory itself.
# Using 'find' is safer than 'rm -rf /tmp/*' in case of strange filenames.
find "${EPHEMERAL_TMP}" -mindepth 1 -delete
rm -rf "${EPHEMERAL_HF_CACHE_DEFAULT_1}"
rm -rf "${EPHEMERAL_HF_CACHE_DEFAULT_2}"
echo "Ephemeral drive cleanup complete."
echo

echo "All specified directories have been cleaned."