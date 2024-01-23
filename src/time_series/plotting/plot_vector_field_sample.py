import os
from copy import deepcopy

import matplotlib as mpl
import numpy as np
import torch
from laplace import Laplace
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from plot import init_figure_third, init_figure_half_column
from torch import nn

from time_series.plotting.plotting_class import PlottingClass
from util.device_setter import train_device


class ODEWrapper(nn.Module):
    def __init__(self, model):
        super(ODEWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(None, x)


class VectorFieldPlotSample(PlottingClass):
    NAME = "vector_field_sample"

    def __init__(self, results_dir):
        super(VectorFieldPlotSample, self).__init__(results_dir=results_dir)
        self.results_dir = results_dir

    def figure(self):
        fig, ax = init_figure_third()
        pos = mpl.transforms.Bbox([[0.15, 0.2], [0.8, 0.95]])
        ax.set_position(pos)
        self.plot_fig(ax, fig)
        self.save_figure(fig=fig)

    def figure_half_column(self):
        fig, ax = init_figure_half_column()
        pos = mpl.transforms.Bbox([[0.25, 0.3], [0.75, 0.95]])
        ax.set_position(pos)
        self.plot_fig(ax, fig)
        self.save_figure(fig=fig)

    def plot_fig(self, ax, fig):
        data = self.load()
        if data is not None:
            x_train, U, V, X, Y, var = data
            mean = np.linalg.norm(np.stack((U, V), axis=-1), axis=-1)
            np_max = np.max(var)
            pc = ax.pcolormesh(
                X, Y, var, alpha=0.5, shading="auto", cmap="Blues", vmin=-0.0, vmax=np_max/5, linewidth=0,
            )
            pc.set_edgecolor('face')
            for i in range(x_train.shape[0]):
                ax.scatter(x_train[i, :, 0], x_train[i, :, 1], s=0.5, color="k")
            print(np.min(var))
            print(np_max)
            ax.quiver(X[::5], Y[::5], U[::5, ::5], V[::5, ::5])
            ax.set_xlabel("$q$")
            ax.set_ylabel("$p$")
            axins = inset_axes(ax,
                               width="5%",
                               height="100%",
                               loc='right',
                               borderpad=-2
                               )
            fig.colorbar(pc, cax=axins, orientation="vertical")

    def plot(self, ax):
        data = self.load()
        if data is not None:
            x_train, U, V, X, Y, var = data
            pc = ax.pcolormesh(
                X, Y, var, alpha=0.5, shading="auto", cmap="Blues", linewidth=0, antialiased=True, rasterized=True
            )
            pc.set_edgecolor('face')
            for i in range(x_train.shape[0]):
                ax.scatter(x_train[i, :, 0], x_train[i, :, 1], s=0.5, color="k")
            ax.quiver(X[::5], Y[::5], U[::5, ::5], V[::5, ::5])
            ax.set_xlabel("$q$")
            ax.set_ylabel("$p$")
            return pc

    def load(self):
        file = os.path.join(self.fig_data, f"{self.NAME}.pt")
        if os.path.exists(file):
            data_dict = torch.load(os.path.join(self.fig_data, f"{self.NAME}.pt"))
            x_train = data_dict["x_train"]
            U = data_dict["U"]
            V = data_dict["V"]
            X = data_dict["X"]
            Y = data_dict["Y"]
            var = data_dict["var"]
            return x_train, U, V, X, Y, var
        return None

    def data(self, la: Laplace, x: torch.Tensor, N: int):
        x_train = x
        X = np.linspace(-0.5, 2.5, N)
        Y = np.linspace(-0.5, 1.5, N)
        U, V = np.meshgrid(X, Y)
        shape_u = U.shape
        U = torch.tensor(
            U, dtype=torch.float32, device=train_device.get_device()
        ).flatten()
        V = torch.tensor(
            V, dtype=torch.float32, device=train_device.get_device()
        ).flatten()
        data = torch.stack((U, V)).transpose(0, 1)
        num_samples = 5000
        la.prior_precision = torch.tensor([50.])
        la._posterior_scale = None
        parameter_sample = la.sample(num_samples)
        sol_list = []

        print(data.shape)
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
            model = la.model.model.odefunc
            f_l = model(None, data)
            sol_list.append(deepcopy(f_l.detach()))
        sol_list = torch.stack(sol_list, dim=-1)
        vec = torch.mean(sol_list, dim=-1)
        var = torch.norm(torch.std(sol_list, dim=-1)**2, dim=-1).cpu().numpy().reshape(shape_u)
        U = vec[:, 0].detach().cpu().numpy().reshape(shape_u)
        V = vec[:, 1].detach().cpu().numpy().reshape(shape_u)
        torch.save(
            {"U": U, "V": V, "X": X, "Y": Y, "var": var, "x_train": x_train},
            os.path.join(self.fig_data, f"{self.NAME}.pt"),
        )
