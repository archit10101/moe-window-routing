# CPU smoke test: tiny model, synthetic data (python scripts/make_synthetic_data.py --out /tmp/synth).
# Finishes in well under a minute on a laptop CPU.
exec(open('config/pilot_base.py').read())

run_name = 'smoke-cpu'
data_dir = '/tmp/synth'
out_root = '/tmp/moe-pilot-runs'

n_layer = 2
n_embd = 64
n_head = 4
block_size = 128
vocab_size = 1024
n_exp = 4
top_k = 2
expert_hidden = 64
shared_hidden = 128
window_layers = [0, 4]

lr = 3e-3
total_tokens = 40 * 64 * 128  # 40 steps of batch_size x block_size

log_interval = 5
eval_interval = 20
eval_batches = 2
final_eval_batches = 4
eval_batch_size = 8
