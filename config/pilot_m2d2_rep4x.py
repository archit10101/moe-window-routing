# M2D2-Wiki-8 uneven design 'rep4x' (docs/M2D2_UNEVEN_PROMPT.md part 5): the even M2D2 recipe on a build
# where one domain (philosophy, or natural_sciences if it runs out) has 4x the training data (60M of ~165M tokens).
# Run configs in config/runs/m2d2r4-*.py exec this.
exec(open('config/pilot_m2d2.py').read())

data_dir = '/home/architta/orcd/scratch/moe-pilot/data_m2d2w8_rep4x'
total_tokens = 165_000_000  # 2,518 steps of 64 x 1,024 tokens
