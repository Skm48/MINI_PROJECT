"""
Shared utilities: config loading, seed setting, device selection, plotting helpers.
"""
import torch
import numpy as np
import random
import yaml


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get best available device (GPU on Colab, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def load_config(path: str = "configs/config.yaml") -> dict:
    """Load YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
