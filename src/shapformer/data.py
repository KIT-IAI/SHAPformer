import json
from typing import Optional

import torch


def load_data(path, device: Optional[str] = None):
    # loads preprocessed samples (dictionaries with "x_enc", "x_dec" and "y") from a JSON file as tensors;
    # device: where to put the tensors, by default the GPU if one is available, else the CPU
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(path, "r") as f:
        data = json.load(f)
    for i in range(len(data)):
        for key in data[i]["x_enc"]:
            data[i]["x_enc"][key] = torch.tensor(data[i]["x_enc"][key], device=device)
        for key in data[i]["x_dec"]:
            data[i]["x_dec"][key] = torch.tensor(data[i]["x_dec"][key], device=device)
        data[i]["y"] = torch.tensor(data[i]["y"], device=device)
    return data
