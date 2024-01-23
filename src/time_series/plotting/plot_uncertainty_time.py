import os

import numpy as np
import torch
from laplace import Laplace
from matplotlib import pyplot as plt
from time_series.models.ode_block import ODEBlock

from data.time_series_data.toy_timeseriers_base import ODE
from data.time_series_data.toy_timeseriers_dataset import TimeSeriesDataset
from time_series.plotting.plotting_class import PlottingClass
from util.device_setter import train_device


class UncertaintyPlot(PlottingClass):

    NAME = "uncertainty"

    def __init__(self, results_dir):
        super(UncertaintyPlot, self).__init__(results_dir=results_dir)

    def plot(self, ax: plt.Axes):
        try:
            t_test, t_train, x_train, f_mu, pred_std, true_solution = self.load()
            ax.set_title("LA")
            rel_var = pred_std
            zeros = np.zeros(np.shape(rel_var)[0])
            ax.fill_between(
                t_test,
                zeros,
                rel_var[:, 0],
                label="$\sqrt{\mathbb{V}\,[q]}$",
                color="tab:blue",
                alpha=0.3,
                lw=0,
            )
            ax.fill_between(
                t_test,
                zeros,
                rel_var[:, 1],
                label="$\sqrt{\mathbb{V}\,[p]}$",
                color="tab:orange",
                alpha=0.3,
                lw=0,
            )
            ax.plot(
                t_test,
                rel_var[:, 0],
                color="tab:blue",
            )
            ax.plot(
                t_test,
                rel_var[:, 1],
                color="tab:orange",
            )
            ax.legend(ncol=2)
            ax.set_ylim([0.0, 1.])

            ax.set_ylabel("$\sqrt{\mathbb{V}\,[q]}, \sqrt{\mathbb{V}\,[p]}$")
            ax.set_xlabel("Time $t$")
        except FileNotFoundError as e:
            print(e)

    def load(self):
        file = os.path.join(self.fig_data, f"{self.NAME}.pt")
        data_dict = torch.load(file)
        x_train = data_dict["x_train"]
        t_train = data_dict["t_train"]
        t_test = data_dict["t_test"]
        f_mu = data_dict["f_mu"]
        pred_std = data_dict["pred_std"]
        true_solution = data_dict["true_solution"]
        return t_test, t_train, x_train, f_mu, pred_std, true_solution

    def data(
        self,
        la: Laplace,
        data: TimeSeriesDataset,
        ode: ODE,
        t_max_test: float,
        steps_t_test: int,
    ):
        x_train, t_train, x0 = data.x, data.t, data.x0
        t_test = torch.linspace(0, t_max_test, steps_t_test)
        x0 = data.x0.to(train_device.get_device())
        la.model.t = t_test
        model = ODEBlock(ode, opts=ode.opts)
        true_solution = model(x0, t_test)[0].detach().cpu().numpy()
        f_mu, f_var = la(x0[0].unsqueeze(0))
        t_test = t_test.flatten().cpu().numpy()
        f_mu = f_mu.squeeze().detach().cpu().numpy().reshape(t_test.shape[0], 2)
        f_sigma = torch.diagonal(f_var.squeeze().sqrt().cpu()).numpy()
        pred_std = np.sqrt(f_sigma ** 2 + la.sigma_noise.item() ** 2).reshape(
            t_test.shape[0], 2
        )
        torch.save(
            {
                "t_train": data.t,
                "t_test": t_test,
                "x_train": x_train[0],
                "f_mu": f_mu,
                "pred_std": pred_std,
                "true_solution": true_solution,
            },
            os.path.join(self.fig_data, f"{self.NAME}.pt"),
        )
