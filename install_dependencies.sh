#!/bin/bash

# This script installs the required Python libraries for the project.

echo "Installing Python dependencies..."

# Using pip to install the libraries
pip install torch numpy matplotlib tiktoken transformers
pip install wandb
pip install tqdm datasets

echo "Installation of dependencies is complete."