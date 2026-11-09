# M2D2-Wiki-8 replication (docs/M2D2_PROMPT.md part 6): the pilot recipe on the M2D2-Wiki-8 data,
# with a token budget matched to its ~120M training tokens. Run configs in config/runs/m2d2-*.py exec this.
exec(open('config/pilot_base.py').read())

data_dir = '/home/architta/orcd/scratch/moe-pilot/data_m2d2w8'
total_tokens = 120_000_000  # 1,832 steps of 64 x 1,024 tokens (random windows; see README_M2D2.md)
eval_interval = 100
