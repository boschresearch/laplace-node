import os
from typing import Union

import torch

from data.find_dataset_class import find_dataset_class
from options.yaml_service import load_yaml
from time_series.models.ode_block import ODEBlock
from time_series.plotting.plot_final_solution import FinalSolutionPlot
from time_series.plotting.plot_loss import LossPlot
from time_series.plotting.plot_vector_field import VectorFieldPlot
from time_series.time_series_options.experiment_options import ExperimentOptions
from time_series.time_series_util import find_network
from util.device_setter import train_device

STEPS_T_TEST = 500


def generate_data_for_plots(results_dir: str, niter: int = None):
    dataset, model, niter, _ = load_model_and_options(results_dir, niter)
    N = 15
    LossPlot(results_dir=results_dir).data(max_iter=niter, experiment_dir=results_dir)
    train_data = dataset.get_dataset("train")
    ode_class = dataset.ode_class.to(train_device.get_device())
    model = model.to(train_device.get_device())
    FinalSolutionPlot(results_dir=results_dir).data(
        data=train_data,
        model=model,
        ode=ode_class,
        t_max_train=dataset.T_MAX_TRAIN,
        t_max_test=dataset.T_MAX_TEST,
        steps_t_test=STEPS_T_TEST,
    )

    if train_data.x.shape[-1] == 2:
        VectorFieldPlot(results_dir=results_dir).data(N=N, model=model, x=train_data.x)


def generate_plots(results_dir: str):
    dataset, _, _, _ = load_model_and_options(
        results_dir, niter=None, return_model=False
    )
    LossPlot(results_dir=results_dir).figure()
    FinalSolutionPlot(results_dir=results_dir).figure()

    train_data = dataset.get_dataset("train")
    if train_data.x.shape[-1] == 2:
        VectorFieldPlot(results_dir=results_dir).figure()


def load_model_and_options(
    results_dir: str, niter: Union[int, None], return_model: bool = True
):
    options_file = os.path.join(results_dir, "aphinity_options", "opts.yaml")
    options_dict = load_yaml(options_file)
    opts = ExperimentOptions(**options_dict)
    train_device.set_device("cpu")
    dataset = find_dataset_class(opts.dataset)()
    if return_model:
        neural_net = find_network(dataset, opts)
        neural_ode = ODEBlock(neural_net, opts.odeint_options)
        if niter is None:
            niter = opts.niter - 1
        state_dict = torch.load(
            os.path.join(results_dir, "checkpoints", f"model_iter_{niter}.pt"),
            map_location=train_device.get_device(),
        )
        neural_ode.load_state_dict(state_dict)

    else:
        neural_ode = None
    return dataset, neural_ode, niter, opts
