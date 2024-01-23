import os
import pathlib
from typing import Generator

import torch
from torch.nn import MSELoss, Module
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from data.find_dataset_class import find_dataset_class
from evaluate_time_series import evaluate_data_and_plot
from time_series.models.ode_block import ODEBlock
from time_series.time_series_options.experiment_options import ExperimentOptions
from time_series.time_series_util import find_network
from util.device_setter import train_device
from util.helper_functions import inf_generator

basedir = os.path.dirname(pathlib.Path().absolute())


def find_optimizer(opts: ExperimentOptions):
    if opts.optimizer == "adam":
        return torch.optim.Adam
    elif opts.optimizer == "sgd":
        return torch.optim.SGD


def iterate_one_training_step(
    iteration: int,
    opts: ExperimentOptions,
    data_generator: Generator,
    optimizer: Optimizer,
    neural_ode: Module,
    loss_function: _Loss,
    loss_log: torch.Tensor,
):
    t, x, x0 = data_generator.__next__()
    x = x.to(train_device.get_device())
    x0 = x0.to(train_device.get_device())
    x0.requires_grad_(True)
    t = t[0].to(train_device.get_device())
    optimizer.zero_grad()
    logits = neural_ode(x0, t)
    loss = loss_function(logits, x)
    loss.backward()
    print(f"Iteraion: {iteration}, Loss:  {loss.item()}")
    loss_log[iteration] = loss.detach().cpu()
    optimizer.step()
    if (iteration + 1) % opts.checkpoint_freq == 0 or (iteration + 1) == opts.niter:
        torch.save(
            logits.detach().cpu(),
            os.path.join(opts.experiment_dir, f"step_{iteration}.pt"),
        )
        torch.save(loss_log, os.path.join(opts.experiment_dir, "loss.pt"))
        torch.save(
            neural_ode.state_dict(),
            os.path.join(
                opts.experiment_dir, "checkpoints", f"model_iter_{iteration}.pt"
            ),
        )


def run(opts: ExperimentOptions):
    torch.cuda.empty_cache()
    torch.manual_seed(opts.random_seed)
    dataset = find_dataset_class(opts.dataset)()
    neural_net = find_network(dataset, opts)
    neural_ode = ODEBlock(neural_net, opts.odeint_options)
    if train_device.get_device() != "cpu":
        neural_ode = neural_ode.cuda()
    optimizer = find_optimizer(opts)
    optimizer = optimizer(neural_ode.parameters(), lr=opts.lr)
    loss_function = MSELoss()
    loss_log = torch.empty(opts.niter)
    data_generator = inf_generator(
        dataset.return_dataloader(
            phase="train",
            batch_size=opts.batch_size,
            shuffle=True,
            num_threads=0
        )
    )
    print("Starting training ...")
    if opts.run_experiment:
        for iteration in range(opts.niter):
            iterate_one_training_step(
                iteration=iteration,
                opts=opts,
                data_generator=data_generator,
                optimizer=optimizer,
                neural_ode=neural_ode,
                loss_function=loss_function,
                loss_log=loss_log,
            )

    print("Finished training")
    print("Plotting results")
    evaluate_data_and_plot(run_folder=opts.experiment_dir, niter=opts.niter - 1, apply_fit_laplace=opts.run_laplace,
                           generate_data=opts.evaluate_experiment)
    print("Finished plotting.")


if __name__ == "__main__":

    opts = ExperimentOptions(options_file="training_opts.yaml")
    run(opts)
