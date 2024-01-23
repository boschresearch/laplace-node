import os

import torch
from laplace import Laplace
from matplotlib import pyplot as plt

from data.time_series_data.toy_timeseriers_base import ODE
from data.time_series_data.toy_timeseriers_dataset import TimeSeriesDataset
from options.options_enum import SolverEnum
from time_series.models.ode_block import ODEBlock
from time_series.plotting.plot_uncertainty_potential import ODEWrapper
from time_series.plotting.plotting_class import PlottingClass
from time_series.time_series_options.experiment_options import ExperimentOptions
from util.device_setter import train_device


class FinalSolutionUncertaintySamplePlot(PlottingClass):

    NAME = "final_solution_uncertainty_sample"

    def __init__(self, results_dir):
        super(FinalSolutionUncertaintySamplePlot, self).__init__(results_dir=results_dir)

    def plot(self, ax: plt.Axes):
        try:
            t_test, t_train, x_train, f_mu, pred_std, true_solution = self.load()
            if true_solution is not None:
                ax.plot(t_test, true_solution[:, 0], "--", color="black", lw=0.5)
                ax.plot(t_test, true_solution[:, 1], "--", color="black", lw=0.5)
            ax.plot(t_train, x_train[:, 0], ".", color="black")
            ax.plot(t_train, x_train[:, 1], ".", color="black")
            ax.plot(t_test, f_mu[:, 0], label="$\mathbb{E}[q]$", color="tab:blue")
            ax.plot(t_test, f_mu[:, 1], label="$\mathbb{E}[p]$", color="tab:orange")
            ax.fill_between(
                t_test,
                f_mu[:, 0] - pred_std[:, 0] * 2,
                f_mu[:, 0] + pred_std[:, 0] * 2,
                alpha=0.3,
                color="tab:blue",
                label="$2\sqrt{\mathbb{V}\,[q]}$",
                lw=0,
            )
            ax.fill_between(
                t_test,
                f_mu[:, 1] - pred_std[:, 1] * 2,
                f_mu[:, 1] + pred_std[:, 1] * 2,
                alpha=0.3,
                color="tab:orange",
                label="$2\sqrt{\mathbb{V}\,[p]}$",
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
        opts: ExperimentOptions,
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
        old_model = la.model
        la.model = ODEWrapper(la.model.model.odefunc)

        def sample_vf(t, x):
            sample = torch.randn(x.shape)
            mu, var = la(x)
            var_mu = (sample @ torch.linalg.cholesky(var) + mu).squeeze(0)
            return var_mu

        opts.odeint_options.solver = SolverEnum.symplectic_euler
        ode_block = ODEBlock(sample_vf, opts.odeint_options)

        r_list = []
        for i in range(1000):
            print(i)
            result = ode_block(x0, t_test)
            r_list.append(result)

        result = torch.stack(r_list, dim=0)
        f_mu = torch.mean(result, dim=0).squeeze(0).detach().cpu().numpy()
        pred_std = torch.std(result, dim=0).squeeze(0).detach().cpu().numpy()

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
        la.model = old_model
