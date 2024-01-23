import torch
from torch.utils import data as data


class TimeSeries:
    def __init__(self, t: torch.Tensor, x: torch.Tensor, x0: torch.Tensor):
        self.t = t
        self.x = x
        self.x0 = x0


class TimeSeriesDataset(data.Dataset):
    def __init__(self, data: TimeSeries):
        self.x = data.x
        self.t = data.t
        self.x0 = data.x0

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.t, self.x[item], self.x0[item]
