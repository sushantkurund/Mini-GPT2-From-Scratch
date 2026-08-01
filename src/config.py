import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data
BATCH_SIZE = 64
BLOCK_SIZE = 128

# Training
MAX_ITERS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 200
LEARNING_RATE = 3e-4

# Model
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2