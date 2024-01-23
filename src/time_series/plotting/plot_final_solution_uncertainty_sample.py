import os
from copy import deepcopy

import torch
from laplace import Laplace
from matplotlib import pyplot as plt

from data.time_series_data.toy_timeseriers_base import ODE
from data.time_series_data.toy_timeseriers_dataset import TimeSeriesDataset
from time_series.models.ode_block import ODEBlock
from time_series.plotting.plotting_class import PlottingClass
from util.device_setter import train_device


class SampleUncertaintyPlot(PlottingClass):

    NAME = "sample_uncertainty"

    def __init__(self, results_dir):
        super(SampleUncertaintyPlot, self).__init__(results_dir=results_dir)
        self.results_dir = results_dir

    def plot(
        self,
        ax: plt.Axes,
        labelx_mean="$\mathbb{E}[q]$",
        labely_mean="$\mathbb{E}[p]$",
        labelx_var="$2\sqrt{\mathbb{V}\,[q]}$",
        labely_var="$2\sqrt{\mathbb{V}\,[p]}$",
    ):
        try:
            t_test, t_train, x_train, f_mu, pred_std, true_solution = self.load()
            if true_solution is not None:
                ax.plot(t_test, true_solution[:, 0], "--", color="black", lw=0.5)
                ax.plot(t_test, true_solution[:, 1], "--", color="black", lw=0.5)
            if x_train is not None:
                ax.plot(t_train, x_train[:, 0], ".", color="black")
                ax.plot(t_train, x_train[:, 1], ".", color="black")
            ax.plot(t_test, f_mu[:, 0], label=labelx_mean, color="tab:blue")
            ax.plot(t_test, f_mu[:, 1], label=labely_mean, color="tab:orange")

            ax.fill_between(
                t_test,
                f_mu[:, 0] - pred_std[:, 0] * 2,
                f_mu[:, 0] + pred_std[:, 0] * 2,
                alpha=0.3,
                color="tab:blue",
                label=labelx_var,
                lw=0,
            )
            ax.fill_between(
                t_test,
                f_mu[:, 1] - pred_std[:, 1] * 2,
                f_mu[:, 1] + pred_std[:, 1] * 2,
                alpha=0.3,
                color="tab:orange",
                label=labely_var,
                lw=0,
            )

            ax.set_ylabel("$q, p$")
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
        num_samples = 1000
        la.prior_precision = torch.tensor([50.])
        la._posterior_scale = None
        parameter_sample = la.sample(num_samples)
        sol_list = []

        for i, param in enumerate(parameter_sample):
            print(i)
            state_dict = torch.load(
                os.path.join(self.results_dir, "checkpoints", f"model_iter_{499}.pt"),
                map_location=train_device.get_device(),
            )
            for key, value in state_dict.items():
                n = torch.numel(value)
                p = param[:n]
                param = param[n:]
                p = p.reshape(value.shape)
                state_dict[key] = p
            neural_ode = la.model
            neural_ode.model.load_state_dict(state_dict)
            model = ODEBlock(neural_ode.model.odefunc, opts=ode.opts)
            sol = model(x0, t_test)[0]
            print(sol[-1])
            sol_list.append(deepcopy(sol.detach()))
        sol_list = torch.stack(sol_list, dim=-1)
        f_mu = torch.mean(sol_list, dim=-1)
        pred_std = torch.std(sol_list, dim=-1)
        model = ODEBlock(ode, opts=ode.opts)
        # true_solution = model(x0, t_test)[0].detach().cpu().numpy()
        true_solution = None
        t_test = t_test.flatten().cpu().numpy()
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
