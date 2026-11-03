"""Small reproducibility helpers shared by the experiment entry points."""
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import yaml


def seed_everything(seed=771):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))


def save_config(cfg, folder):
    with open(Path(folder) / 'config.yaml', 'w') as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)


def append_metrics(folder, metrics):
    with open(Path(folder) / 'history.jsonl', 'a') as stream:
        stream.write(json.dumps(metrics, allow_nan=False) + '\n')


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()
