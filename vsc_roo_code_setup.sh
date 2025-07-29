#!/bin/bash

# ===================================================================================
# SCRIPT TO QUICKLY INSTALL VS CODE EXTENSIONS ON A NEW RUNPOD INSTANCE
# ===================================================================================
#
# HOW TO PREPARE THIS SCRIPT FOR THE FIRST TIME:
#
# STEP 1: DOWNLOAD THE EXTENSION FILE (.VSIX) ON YOUR LOCAL COMPUTER
# -----------------------------------------------------------------------------------
#   a. Go to Extensions tab in Visual Studio Code or in the Visual Studio Marketplace in your web browser.
#   b. Search for the extension you need (e.g., "Roo Code").
#   c. On the extension's page, look on the right-hand side under "Resources" or the "Settings (cogwheel)" icon
#      and click the "Download Extension" link.
#   d. This will download a file with a `.vsix` extension (e.g.,
#      'rooveterinaryinc.roo-cline-3.23.15.vsix'). Save it somewhere you can find it.
#
# STEP 2: UPLOAD THE .VSIX FILE TO YOUR PERSISTENT RUNPOD WORKSPACE
# -----------------------------------------------------------------------------------
#   a. Start a RunPod instance and connect to it with VS Code (Remote - SSH).
#   b. In the VS Code Explorer (the file panel on the left), navigate to your
#      persistent `/workspace/` directory.
#   c. Drag the `.vsix` file from your local computer's file manager and drop it
#      directly into the `/workspace/` folder (or a subfolder, `/workspace/vsc_stuff`) in VS Code.
#   d. Update the path in the command below to match where you put the file.
#
# STEP 3: PREPARE AND RUN THIS SCRIPT
# -----------------------------------------------------------------------------------
#   a. Place this script file (e.g., `vsc_roo_code_setup.sh`) e.g. in your `/workspace/`.
#   b. Make it executable ONE TIME ONLY. Open a terminal and run:
#      chmod +x /workspace/vsc_roo_code_setup.sh
#   c. From now on, every time you start a new RunPod session, just open a
#      terminal and run this script to install your extension:
#      /workspace/vsc_roo_code_setup.sh
#
# ===================================================================================
# --- SCRIPT EXECUTION STARTS HERE ---

# Find the VS Code Server executable path dynamically
CODE_EXECUTABLE=$(find /root/.vscode-server -type f -name code | head -n 1)

# Check if we found the executable
if [ -z "$CODE_EXECUTABLE" ]; then
    echo "Error: VS Code executable not found."
    exit 1
fi

# Run the installation command using the found path
"$CODE_EXECUTABLE" --install-extension /workspace/vsc_stuff/rooveterinaryinc.roo-cline-3.23.15.vsix

echo "Roo Code extension installation command executed."