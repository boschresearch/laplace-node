import os

import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
from torch import nn

from data.base_dataset import BaseDataset
from data.time_series_data.toy_timeseriers_dataset import TimeSeriesDataset, TimeSeries
from time_series.models.ode_block import ODEBlock


class ODE(nn.Module):
    def __init__(self):
        super(ODE, self).__init__()
        pass

    def forward(self, t, x: torch.Tensor):
        pass


class DampedODE(nn.Module):
    def __init__(self, ode):
        super(DampedODE, self).__init__()
        self.ode = ode
        self.damping_constant = 0.2
        self.opts = ode.opts
        self.hamiltonian = ode.hamiltonian
        self.x0 = ode.x0

    def forward(self, t, x):
        q, p = torch.chunk(x, 2, -1)
        q_dot, p_dot = torch.chunk(self.ode(t, x), 2, -1)

        p_dot = p_dot - self.damping_constant * p
        return torch.cat((q_dot, p_dot), dim=-1)


class TimeSeriesToyBase(BaseDataset):
    BATCH_SIZE: int
    T_MAX_TRAIN: float
    T_MAX_TEST: float
    N_SAMPLES_TRAIN: int
    N_SAMPLES_TEST: int
    DIMENSION = 2
    DATA_NOISE: float

    def __init__(self):
        super(TimeSeriesToyBase, self).__init__()
        self.dataset_name = None
        self.has_hamiltonian = False
        self.ode_class = ODE()

    def get_dataset(self, split):

        data_dir = os.path.join(self.path_to_data, self.dataset_name)
        if not os.path.exists(data_dir):
            self.generate_and_save_data()

        if split == "train":
            dataset = self.load_data_train()
        else:
            dataset = self.load_data_test()
        return dataset

    def load_data_train(self) -> data.Dataset:
        data = torch.load(
            os.path.join(self.path_to_data, self.dataset_name, "train_data.pt")
        )
        dataset = TimeSeriesDataset(data)
        return dataset

    def load_data_test(self) -> data.Dataset:
        data = torch.load(
            os.path.join(self.path_to_data, self.dataset_name, "test_data.pt")
        )
        dataset = TimeSeriesDataset(data)
        return dataset

    def dataset_generation(
        self, N: int = 1000, t_max: float = 50
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        ode = self.ode_class
        t, _ = torch.sort(torch.rand(N) * t_max + 1e-8)
        model = ODEBlock(ode, opts=ode.opts)
        out = model(ode.x0, t).detach()
        noise = torch.normal(torch.zeros(out.shape), self.DATA_NOISE)
        return t, (out + noise), ode.x0

    def generate_and_save_data(self,):
        data_dir = os.path.join(self.path_to_data, self.dataset_name)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        t_max = self.T_MAX_TRAIN
        N = self.N_SAMPLES_TRAIN
        dataset_split = "train_data"
        self.make_data(N, data_dir, dataset_split, t_max)

        t_max = self.T_MAX_TEST
        N = self.N_SAMPLES_TEST
        dataset_split = "test_data"
        self.make_data(N, data_dir, dataset_split, t_max)

    def make_data(self, N: int, data_dir: str, dataset_split: str, t_max: float):
        t_train, x_train, x0_train = self.dataset_generation(N=N, t_max=t_max)
        train_set = TimeSeries(x=x_train, t=t_train, x0=x0_train)
        file_dir = os.path.join(data_dir, "%s.pt" % dataset_split)
        torch.save(train_set, open(file_dir, "wb"))
        self.plot_dataset(data_dir, dataset_split, t_train, x_train)

    def plot_dataset(self, data_dir, dataset_split, t_train, x_train):
        for i in range(x_train.shape[0]):
            plt.plot(
                t_train, x_train[i, :, 0], ".", color="tab:blue", alpha=0.5, mew=0.0
            )
            plt.plot(
                t_train, x_train[i, :, 1], ".", color="tab:orange", alpha=0.5, mew=0.0
            )
        plt.savefig(os.path.join(data_dir, "%s.png" % dataset_split), dpi=500)
        plt.clf()
        for i in range(x_train.shape[0]):
            plt.scatter(
                x_train[i, :, 0],
                x_train[i, :, 1],
                color="tab:blue",
                alpha=0.5,
                edgecolors="none",
            )
        plt.savefig(os.path.join(data_dir, "%s_2d.png" % dataset_split), dpi=500)
        plt.clf()


