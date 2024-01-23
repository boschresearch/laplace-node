import os

import torch
from torch import nn

from aphynity.aphinity_options.aphinityoptions import AphinityOptions, DatasetEnum
from aphynity.plotting.final_solution_uncertainty import (
    AphinityFinalSolutionUncertaintyPlot,
)
from aphynity.plotting.imshow_plotting import ImshowPlot
from aphynity.plotting.plot_final_solution import AphinityFinalSolutionPlot
from aphynity.plotting.plot_imshow_uncertainty import ImshowUncertaintyPlot
from aphynity.plotting.vector_field_uncertainty import (
    AphinityVectorFieldPlotUncertainty,
)
from time_series.laplace_torch import fit_laplace
from util.device_setter import train_device

"""
The following code is adapted from APHYNITY Commit 1b0fcfc7
( https://github.com/yuan-yin/APHYNITY
Copyright (c) 2021 Yuan Yin, Vincent Le Guen, Jérémie Dona, licensed under the MIT License,
cf. 3rd-party-license.txt file in the root directory of this source tree)
"""


class ReducedOutputModel(nn.Module):
    def __init__(self, net, si):
        super(ReducedOutputModel, self).__init__()
        self.net = net
        self.si = si

    def forward(self, x, t):
        x = self.net(x, t)
        return x[:, :, :, :: self.si, :: self.si]


def evaluate(
    opts: AphinityOptions,
    net,
    train_data,
    do_fit_laplace: bool = True,
    do_generate_data: bool = True,
):
    if opts.dataset == DatasetEnum.pendulum:
        evaluate_pendulum(do_fit_laplace, do_generate_data, net, opts, train_data)
    if opts.dataset == DatasetEnum.wave:
        evaluate_wave(net, opts, train_data, do_generate_data, do_fit_laplace)


def evaluate_wave(net, opts, train_data, do_generate_data, do_fit_laplace):
    t, x, x0, net = load(net, opts)
    with torch.no_grad():
        plot = ImshowPlot(opts.experiment_dir)
        if do_generate_data:
            plot.data(model=net, x0=x0, t=t, train_data=train_data)
        plot.figure()
        t_interval = 5
        batch_interval = 5
        shape_interval = 8
        t_l = t[::t_interval]
        x_l = x[::batch_interval, :, ::t_interval, ::shape_interval, ::shape_interval]
        x0_l = x0[::batch_interval]
        net = ReducedOutputModel(net, shape_interval)
        print("laplace")
        if do_fit_laplace:
            with torch.enable_grad():
                fit_laplace(
                    results_dir=opts.experiment_dir,
                    model=net,
                    batch_size=1,
                    t=t_l,
                    x0=x0_l,
                    x_train=x_l,
                )
        ind = 0
        plot = ImshowUncertaintyPlot(opts.experiment_dir)
        plot.NAME += f"_{ind}"
        if do_generate_data:
            la = torch.load(os.path.join(opts.experiment_dir, "la.pt"))
            la._device = train_device.get_device()
            plot.data(la=la, train_data=train_data, x0=x0, t=t, index=ind)
        plot.figure()

        ind = 160
        plot = ImshowUncertaintyPlot(opts.experiment_dir)
        plot.NAME += f"_{ind}"
        if do_generate_data:
            la = torch.load(os.path.join(opts.experiment_dir, "la.pt"))
            la._device = train_device.get_device()
            plot.data(la=la, train_data=train_data, x0=x0, t=t, index=ind)
        plot.figure()


def evaluate_pendulum(do_fit_laplace, do_generate_data, net, opts, train_data):
    t, x, x0, net = load(net, opts)
    plot = AphinityFinalSolutionPlot(results_dir=opts.experiment_dir)
    print(opts.experiment_dir)
    if do_generate_data:
        plot.data(
            train_data,
            t_train=t,
            t_test=t,
            x0=x0,
            x_train=x,
            x_test=x,
            model=net,
            index=0,
        )
    plot.figure()
    print("finished plotting")
    if do_fit_laplace:
        fit_laplace(
            results_dir=opts.experiment_dir,
            model=net,
            batch_size=1,
            t=t,
            x0=x0,
            x_train=x,
        )
    la = torch.load(os.path.join(opts.experiment_dir, "la.pt"))
    la._device = train_device.get_device()
    ####################################################
    # Shenanigans to evaluate uncertainty for parameters
    ####################################################
    # H = torch.tensor(la.H.detach(), dtype=torch.float64)
    # mean = la.mean.detach()
    # jitter = torch.diag(torch.ones(H.shape[0], dtype=torch.float64)) * 1e-8
    # H = H + jitter
    # H_inv = torch.inverse(H)
    # print(f"Param: {mean[1]} with Uncertainty {H_inv[1, 1]}")

    plot = AphinityVectorFieldPlotUncertainty(results_dir=opts.experiment_dir)
    if do_generate_data:
        plot.data(la=la, x=x, N=65)
    plot.figure()

    plot = AphinityFinalSolutionUncertaintyPlot(results_dir=opts.experiment_dir)
    if do_generate_data:
        plot.data(la=la, train_data=train_data, x=x, t=t, x0=x0, index=0)
    plot.figure(y_lim=[-2, 8])
    plot.NAME = plot.NAME + "_3"
    if do_generate_data:
        plot.data(la=la, train_data=train_data, x=None, t=t, x0=x0, index=10)
    plot.figure(y_lim=[-2, 8])
    plot.NAME = plot.NAME + "_2"
    x0 = torch.tensor([[0.0, 3.0]])
    if do_generate_data:
        plot.data(la=la, train_data=train_data, x=None, t=t, x0=x0, index=0)
    plot.figure(y_lim=[-3, 35])


def load(net, opts):
    data = torch.load(os.path.join(opts.experiment_dir, "data.pt"))
    x = data["x_train"]
    t = data["t_train"]
    x0 = data["x0"]
    state_dict = torch.load(
        os.path.join(
            opts.experiment_dir, "checkpoints", f"model_iter_{opts.nepoch}.pt"
        ),
        map_location=train_device.get_device(),
    )
    net.load_state_dict(state_dict)
    return t, x, x0, net
