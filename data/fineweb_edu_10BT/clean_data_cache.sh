#!/bin/bash

# This script cleans up data caches from previous runs.
# 'set -e' ensures the script will exit immediately if any command fails.
set -e

echo "--- Cleaning up previous runs from the network drive... ---"
NETWORK_DRIVE_BASE="/workspace"
rm -rf "${NETWORK_DRIVE_BASE}/hf_cache"
rm -rf "${NETWORK_DRIVE_BASE}/tmp" # This is now our general-purpose temp dir
rm -rf "${NETWORK_DRIVE_BASE}/nanoGPTKarpathyFork/data/fineweb_edu_10BT/tokenized_dataset_cache"
rm -f  "${NETWORK_DRIVE_BASE}/nanoGPTKarpathyFork/data/fineweb_edu_10BT"/*.bin
echo "--- Cleanup complete. ---"