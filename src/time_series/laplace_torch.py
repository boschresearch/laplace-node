import os
from pathlib import Path
from typing import Union

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from extend_laplace_torch.torch_hessian_laplace import ManualGGN, FullLaplace2
from time_series.plotting.plot_final_solution_uncertainty import (
    FinalSolutionUncertaintyPlot,
)
from time_series.plotting.plot_training import load_model_and_options
from time_series.plotting.plot_uncertainty_time import UncertaintyPlot
from time_series.plotting.plot_vector_field_uncertainty import (
    VectorFieldPlotUncertainty, VectorFieldNormPlotUncertainty,
)
from util.device_setter import train_device

STEPS_T_TEST = 200


class TimeSeriesModel(nn.Module):
    def __init__(self, model, t):
        super(TimeSeriesModel, self).__init__()
        self.model = model
        self.t = t

    def forward(self, x0):
        out = self.model(x0, self.t).flatten(1, -1)
        return out


def fit_laplace(
    results_dir: str, model, t, x0, x_train, batch_size: int,
):
    model = TimeSeriesModel(model=model, t=t).to(train_device.get_device())
    train_loader = DataLoader(TensorDataset(x0, x_train.flatten(1, -1)), batch_size=batch_size)
    n_epochs = 1000
    la = FullLaplace2(model, "regression", backend=ManualGGN)
    la.fit(train_loader)
    log_prior, log_sigma = (
        torch.ones(1, requires_grad=True),
        torch.ones(1, requires_grad=True),
    )
    hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
    print("optimizing hyperpameters")
    for i in range(n_epochs):
        print(f"Epoch: {i}")
        hyper_optimizer.zero_grad()
        neg_marglik = -la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
        neg_marglik.backward()
        hyper_optimizer.step()
    torch.save(la, os.path.join(results_dir, "la.pt"))
    return la


def generate_data_for_plots_laplace(
    results_dir: str,
    niter: int,
    apply_fit_laplace: bool = True,
    batch_size: Union[int, None] = 1,
):
    """

    Args:
        results_dir:
        niter:
        batch_size: defines the batch size for hyper-parameter tuning for last layer laplace. Note that a large batch size
        can lead to very large Jacobians.
    """
    dataset, neural_ode, niter, opts = load_model_and_options(
        results_dir=results_dir, niter=niter
    )
    x0 = dataset.get_dataset("train").x0
    t = dataset.get_dataset("train").t
    x_train = dataset.get_dataset("train").x
    if apply_fit_laplace:
        fit_laplace(
            results_dir=results_dir,
            model=neural_ode,
            batch_size=batch_size,
            t=t,
            x0=x0,
            x_train=x_train,
        )

    la = torch.load(os.path.join(results_dir, "la.pt"))
    la._device = train_device.get_device()
    generate_data(dataset, la, results_dir)


def generate_plots_laplace(results_dir):
    plot = UncertaintyPlot(results_dir=results_dir)
    plot.figure()
    plot = FinalSolutionUncertaintyPlot(results_dir=results_dir)
    plot.figure()
    plot = VectorFieldPlotUncertainty(results_dir=results_dir)
    plot.figure()
    plot = VectorFieldNormPlotUncertainty(results_dir=results_dir)
    plot.figure()


def generate_data(dataset, la, results_dir):
    plot = VectorFieldPlotUncertainty(results_dir=results_dir)
    plot.data(N=65, la=la, x=dataset.get_dataset("train").x)

    plot = FinalSolutionUncertaintyPlot(results_dir=results_dir)
    plot.data(
        la,
        dataset.get_dataset("train"),
        ode=dataset.ode_class,
        t_max_test=dataset.T_MAX_TEST,
        steps_t_test=STEPS_T_TEST,
    )
