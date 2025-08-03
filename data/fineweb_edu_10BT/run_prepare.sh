#!/bin/bash

# This is the definitive launch script. It will solve the storage problem.
# 'set -e' ensures the script will exit immediately if any command fails.
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "--- Starting Robust Data Preparation ---"

# --- PART 1: CLEANUP ---
echo "[Step 1/3] Cleaning up previous runs..."
"${SCRIPT_DIR}/clean_data_cache.sh"

# --- PART 2: SETUP ENVIRONMENT ---
echo "[Step 2/3] Setting up environment..."
source "${SCRIPT_DIR}/setup_environment.sh"

# --- PART 3: EXECUTE THE PYTHON SCRIPT ---
echo "[Step 3/3] Running the Python preparation script..."
python "${SCRIPT_DIR}/prepare_final.py"

echo "--- All steps completed successfully! ---"