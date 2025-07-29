import wandb
import random
import time
import os

# --- Configuration ---
# Ensure wandb is in offline mode
os.environ["WANDB_MODE"] = "offline"


# Step 1: Initialize the W&B run
wandb.init(project="my-first-project", name="dummy-run")

# Step 2: Log configuration/hyperparameters (optional but good practice)
config = {
    "epochs": 10,
    "learning_rate": 0.001,
    "batch_size": 32
}
wandb.config.update(config)

# Step 3: Dummy training loop with logging
for epoch in range(config["epochs"]):
    loss = random.uniform(0.2, 1.0)+epoch*0.1  # Dummy loss
    accuracy = random.uniform(0.5, 1.0)  # Dummy accuracy

    # Step 4: Log metrics
    wandb.log({"epoch": epoch, "loss": loss, "accuracy": accuracy})

    time.sleep(0.5)  # Simulate time taken per epoch

# Step 5: Finish the run
wandb.finish()
