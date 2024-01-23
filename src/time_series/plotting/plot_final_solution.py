import os

import matplotlib.pyplot as plt
import torch

from data.time_series_data.toy_timeseriers_base import ODE
from time_series.models.ode_block import ODEBlock

from time_series.plotting.plotting_class import PlottingClass
from util.device_setter import train_device


class FinalSolutionPlot(PlottingClass):
    NAME = "final_solution"

    def __init__(self, results_dir):
        super(FinalSolutionPlot, self).__init__(results_dir=results_dir)

    def plot(self, ax: plt.Axes):
        t, t_train, t_max_train, x_train, f, true_solution = self.load()
        ax.axvspan(0.0, t_max_train, alpha=0.2, color="black")
        ax.plot(t, true_solution[:, 0], "--", color="black")
        ax.plot(t, true_solution[:, 1], "--", color="black")
        ax.plot(t_train, x_train[:, 0], ".", color="black")
        ax.plot(t_train, x_train[:, 1], ".", color="black")
        ax.plot(t, f[:, 0], color="red", lw=1)
        ax.plot(t, f[:, 1], color="blue", lw=1)
        ax.set_xlabel("Time t")
        ax.set_ylabel("x1, x2")

    def load(self):
        data_dict = torch.load(os.path.join(self.fig_data, f"{self.NAME}.pt"))
        t_test = data_dict["t_test"]
        t_train = data_dict["t_train"]
        t_max_train = data_dict["t_max_train"]
        x_train = data_dict["x_train"]
        f = data_dict["f"]
        true_solution = data_dict["true_solution"]
        return t_test, t_train, t_max_train, x_train, f, true_solution

    def data(self, data, model, ode: ODE, t_max_train: float, t_max_test: float, steps_t_test: int):
        x_train, t_train, x0 = data.x, data.t, data.x0
        t_test = torch.linspace(0, t_max_test, steps_t_test)
        x0 = x0.requires_grad_(True).to(train_device.get_device())
        result = model(x0, t_test).detach().cpu().numpy()[0]
        model = ODEBlock(ode, opts=ode.opts).to(train_device.get_device())
        true_solution = model(x0, t_test)[0].detach().cpu().numpy()
        x_train = x_train[0]
        torch.save(
            {
                "t_test": t_test,
                "t_train": t_train,
                "t_max_train": t_max_train,
                "x_train": x_train,
                "f": result,
                "true_solution": true_solution
            },
            os.path.join(self.fig_data, f"{self.NAME}.pt"),
        )
