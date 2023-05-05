import torch

STEP_COUNT = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" if torch.backends.mps.is_available() else "cpu"