import jax.random as random
import numpy as np
import torch

from mcmc_time_series.options.mcmc_options import MCMCOptions
from mcmc_time_series.plotting.plot_uncertainty import MCMCFinalSolutionUncertaintyPlot
from mcmc_time_series.plotting.plot_vector_field import MCMCVectorFieldPlotUncertainty


def evaluate(
    model,
    x: np.ndarray,
    x0: np.ndarray,
    opts: MCMCOptions,
    rng_key_predict,
    samples,
    t: np.ndarray,
    t_test: np.ndarray,
):
    vmap_args = (
        samples,
        random.split(rng_key_predict, opts.num_samples*opts.num_chains),
    )
    plot = MCMCFinalSolutionUncertaintyPlot(results_dir=opts.experiment_dir)
    plot.data(model=model, 
        vmap_args=vmap_args, t=t, t_test=t_test, x0=x0, x_train=x, hidden_dim=opts.hidden_dim, index=0
    )
    plot.figure(y_lim=[-2, 12])

    plot = MCMCFinalSolutionUncertaintyPlot(opts.experiment_dir)
    plot.NAME = plot.NAME + "_3"
    plot.data(model=model,
        vmap_args=vmap_args, t=t, t_test=t_test, x0=x0, x_train=x, hidden_dim=opts.hidden_dim, index=10
    )
    plot.figure(y_lim=[-2, 2.5])

    plot = MCMCFinalSolutionUncertaintyPlot(opts.experiment_dir)
    plot.NAME = plot.NAME + "_2"
    x0 = np.array([[0., 3.]])
    plot.data(model=model,
        vmap_args=vmap_args, t=t, t_test=t_test, x0=x0, x_train=x, hidden_dim=opts.hidden_dim, index=0, use_x_train=False
    )
    plot.figure(y_lim=[-3, 35])
    plot = MCMCVectorFieldPlotUncertainty(opts.experiment_dir)
    plot.data(x_train=x, N=65, hidden_dim=opts.hidden_dim, vmap_args=vmap_args, batch_size = opts.batch_size, use_aug_model=opts.use_aug_dynamics)
    plot.figure()
