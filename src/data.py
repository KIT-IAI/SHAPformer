import json
import torch


def load_data(path, device="cuda"):
    with open(path, "r") as f:
        data = json.load(f)
    for i in range(len(data)):
        for key in data[i]["x_enc"]:
            data[i]["x_enc"][key] = torch.tensor(data[i]["x_enc"][key], device=device)
        for key in data[i]["x_dec"]:
            data[i]["x_dec"][key] = torch.tensor(data[i]["x_dec"][key], device=device)
        data[i]["y"] = torch.tensor(data[i]["y"], device=device)
    return data
