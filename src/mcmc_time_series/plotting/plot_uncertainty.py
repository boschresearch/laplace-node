import sys

from aphynity.datasets import DampledPendulum
from data.time_series_data.lotka_volterra_dataset import LotkaVolterraODE
from time_series.models.ode_block import ODEBlock

if sys.platform == "linux":
    from jax._src.api import vmap
    import jax.numpy as jnp
    from mcmc_time_series.models.model import bnn_odenet, predict
import matplotlib.pyplot as plt
import numpy as np
from time_series.plotting.plot_final_solution_uncertainty import (
    FinalSolutionUncertaintyPlot,
)
import torch
import os


class MCMCFinalSolutionUncertaintyPlot(FinalSolutionUncertaintyPlot):

    NAME = "final_solution_uncertainty"

    def __init__(self, results_dir):
        super(MCMCFinalSolutionUncertaintyPlot, self).__init__(results_dir=results_dir)

    def plot(self, ax: plt.Axes):
        try:
            t_test, t_train, x_train, f_mu, pred_std, true_solution = self.load()
            model = DampledPendulum(path="", num_seq=1, time_horizon=20, dt=0.1)
            # model = ODEBlock(ode, opts=ode.opts)
            # x0 = ode.x0
            # true_solution = model.true_solution(torch.tensor(t_test), x_train[0])
            # if true_solution is not None:
            #     ax.plot(t_test, true_solution[0, :, 0], "--", color="black", lw=0.5)
            #     ax.plot(t_test, true_solution[0, :, 1], "--", color="black", lw=0.5)
            if x_train is not None:
                ax.plot(t_train, x_train[0, :, 0], ".", color="black")
                ax.plot(t_train, x_train[0, :, 1], ".", color="black")
            ax.plot(t_test, f_mu[0, :, 0], label="$\mathbb{E}[x]$", color="tab:blue")
            ax.plot(t_test, f_mu[0, :, 1], label="$\mathbb{E}[y]$", color="tab:orange")
            print(pred_std.shape)
            ax.fill_between(
                t_test,
                pred_std[0, 0, :, 0],
                pred_std[1, 0, :, 0],
                alpha=0.3,
                color="tab:blue",
                label="$2\sqrt{\mathbb{V}\,[x]}$",
                lw=0,
            )
            ax.fill_between(
                t_test,
                pred_std[0, 0, :, 1],
                pred_std[1, 0, :, 1],
                alpha=0.3,
                color="tab:orange",
                label="$2\sqrt{\mathbb{V}\,[y]}$",
                lw=0,
            )

            ax.set_ylabel("$q, p$")
            ax.set_xlabel("Time $t$")
        except FileNotFoundError as e:
            print(e)

    def data(self, model, vmap_args, t, t_test, x0, x_train, hidden_dim, index: int, use_x_train = True):

        x0 = x0[index][None]
        predictions = vmap(
            lambda samples, rng_key: predict(
                model, rng_key, samples, t_test, x0, hidden_dim=hidden_dim
            )
        )(*vmap_args)

        # compute mean prediction and confidence interval around median
        mean_prediction = jnp.mean(predictions, axis=0)
        percentiles = np.percentile(predictions, [2.275, 97.725], axis=0)
        x_train = x_train[index][None]
        if not use_x_train:
            x_train = None

        torch.save(
            {
                "t_train": t,
                "t_test": t_test,
                "x_train": x_train,
                "f_mu": mean_prediction,
                "pred_std": percentiles,
                "true_solution": None,
            },
            os.path.join(self.fig_data, f"{self.NAME}.pt"),
        )
