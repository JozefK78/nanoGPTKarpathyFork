### To initialize wandb online connection before wandb logging the training run
```
wandb login [API key from my wandb accoun] 
```

### ```nohup``` run of torchrun for large training runs:

So as the training wouldn't stop if the terminal window is closed or lost.

```nohup torchrun --standalone --nproc_per_node=7 train.py config/train_fineweb_10BT.py > training.log 2>&1 &```

without ```wandb``` login

```nohup env WANDB_MODE=offline torchrun --standalone --nproc_per_node=2 train.py config/train_fineweb_10BT.py > training.log 2>&1 &```


### Setup for no api request at each wandb run

Certainly. Below is a **full sequence of commands** to persist and reuse your Weights & Biases (`wandb`) API key across multiple RunPod pods using a **shared network drive** (e.g. mounted at `/workspace`).

---



### Step-by-step: Reuse `wandb` API Key Across Pods: In the **first pod** (after logging in):

```bash
# 1. Login to wandb (you probably already did this)
wandb login [API key]

# 2. Copy the generated .netrc file from /root to your shared /workspace drive
cp /root/.netrc /workspace/.netrc
```

---

### In **each new pod** where you want to reuse the login:

```bash
# 1. Copy the saved .netrc from the shared drive back into /root
cp /workspace/.netrc /root/.netrc

# 2. Set secure permissions (important!)
chmod 600 /root/.netrc

# 3. (Optional) Confirm it works — should say you're logged in
wandb login --verify
```

---

### Optional: Automate it in `bash` (to put in your pod setup script)

```bash
# Only copy if .netrc doesn't already exist
if [ ! -f /root/.netrc ] && [ -f /workspace/.netrc ]; then
  cp /workspace/.netrc /root/.netrc
  chmod 600 /root/.netrc
fi
```

---
### Sampling model with sample inferences
Once your training has produced a checkpoint file in the out-fineweb-10BT directory, you can use the following command to generate the three different completions for your prompt:

```python sample.py --out_dir=out-fineweb-10BT --start="I am Macbeth, therefore" --num_samples=3```


```python sample.py --out_dir=out-fineweb-10BT --start="I am happy beacause" --num_samples=3 --seed=42 --init_from=ckpt_6kiter_weight_decay_0_1.pt```

```ps aux | grep train.py```

```kill XXX```