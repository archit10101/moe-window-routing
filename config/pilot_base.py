# Defaults for every pilot run (docs/PROMPT.md part 4).
# Run configs in config/runs/ exec this file and override only what differs.
# micro_batch_size lives only here: change it here (e.g. to 16) after the memory check.

# run / paths
run_name = None  # required
out_root = '/home/architta/orcd/scratch/moe-pilot/runs'
data_dir = '/home/architta/orcd/scratch/moe-pilot/data'
seed = 0  # controls init and data order

# model (real pilot shape; smoke-cpu overrides with the tiny test model)
n_layer = 8
n_embd = 512
n_head = 8
block_size = 1024
vocab_size = 50304
n_exp = 8
top_k = 2
expert_hidden = 512
shared_hidden = 1024
window_layers = [0] * 8  # window arm: [0, 0, 0, 0, n, n, n, n]

# optimization
lr = None  # required: peak learning rate
min_lr_ratio = 0.1
warmup_frac = 0.02
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
batch_size = 64
micro_batch_size = 64
total_tokens = None  # required
aux_coef = 0.01
z_coef = 0.001

# logging / eval / checkpoints
log_interval = 50
eval_interval = 500
eval_batches = 20
final_eval_batches = 100
eval_batch_size = 16
ckpt_interval_minutes = 15

# system
device = 'auto'  # 'auto' = cuda if available, else cpu
dtype = 'bfloat16'  # autocast on CUDA; plain fp32 on CPU
num_threads = 4
benchmark_steps = 0
resume = True
