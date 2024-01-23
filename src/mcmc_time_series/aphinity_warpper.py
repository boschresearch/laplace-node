import os
from pathlib import Path

import torch

from aphynity.datasets import init_dataloaders

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_aphinity_dataset(dataset: str):
    data_path = Path(__file__).parent.parent / "datasets"
    train, test, train_data = init_dataloaders(dataset, data_path)
    x = []
    for i in range(train_data.len):
        x.append(train_data[i]["states"])
        t = train_data[i]["t"]
    x = torch.stack(x, dim=0)
    x0 = x[:, :, 0]
    t_max_test = torch.max(t).item() * 4
    t_test = torch.linspace(0, t_max_test, 200)
    return x0, x.transpose(1, 2), t, t_test


