import os

import torch
from matplotlib import pyplot as plt

from time_series.plotting.plotting_class import PlottingClass


class LossPlot(PlottingClass):
    NAME = "loss"

    def __init__(self, results_dir):
        super(LossPlot, self).__init__(results_dir=results_dir)

    def plot(self, ax: plt.Axes):
        loss, max_iter = self.load()
        ax.semilogy(loss[:max_iter], color="green")
        # ax.set_ylim(1e-5, 10.0)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")

    def load(self):
        data_dict = torch.load(os.path.join(self.fig_data, f"{self.NAME}.pt"))
        loss = data_dict["loss"]
        max_iter = data_dict["max_iter"]
        return loss, max_iter

    def data(self, experiment_dir, max_iter):
        loss = torch.load(os.path.join(experiment_dir, "loss.pt"))
        data_dict = {"loss": loss, "max_iter": max_iter}
        torch.save(data_dict, os.path.join(self.fig_data, f"{self.NAME}.pt"))
