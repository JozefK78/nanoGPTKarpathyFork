# config for training GPT-2 (124M) on a sample of FineWeb-Edu
# This is a sample run to test the infrastructure on a single GPU.

out_dir = 'out-fineweb-1BT'
eval_interval = 200
log_interval = 10
always_save_checkpoint = True
eval_iters = 200

# wandb logging
wandb_log = True
wandb_project = 'fineweb'
wandb_run_name = 'gpt2-fineweb-sample'

# data
dataset = 'fineweb_edu_1BT_sample'
# total batch size is ~0.5M tokens
total_batch_size = 524288
B = 8 # if memory becomes an issue, lower this
T = 1024
# gradient_accumulation_steps is calculated automatically in train.py

# model
# n_layer, n_head, n_embd are not set here, so they will default to GPT-2 (124M) params in model.py
# dropout = 0.0 # for pretraining

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 1000 # total number of training iterations (kept at 1000 as requested)
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 100 # Changed to match train.py default
lr_decay_iters = 1000 # Set to max_iters as requested for testing phase
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# debug
debug_batches = False