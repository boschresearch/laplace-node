import os

import numpy as np
import torch
from laplace import Laplace
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from torch import nn

from plot import init_figure_third, plt, init_figure_half_column
from time_series.plotting.plotting_class import PlottingClass
from util.device_setter import train_device

import matplotlib as mpl


class ODEWrapper(nn.Module):
    def __init__(self, model):
        super(ODEWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(None, x)


class VectorFieldPlotUncertainty(PlottingClass):
    NAME = "vector_field_uncertainty"

    def __init__(self, results_dir):
        super(VectorFieldPlotUncertainty, self).__init__(results_dir=results_dir)

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
            x_train, U, V, X, Y, std = data
            mean = np.linalg.norm(np.stack((U, V), axis=-1), axis=-1)
            np_max = np.max(std)
            pc = ax.pcolormesh(
                X, Y, std, alpha=0.5, shading="auto", cmap="Blues", vmin=-0.0, vmax=0.05, linewidth=0,
            )
            pc.set_edgecolor('face')
            for i in range(x_train.shape[0]):
                ax.scatter(x_train[i, :, 0], x_train[i, :, 1], s=0.5, color="k")
            print(np.min(std))
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
            x_train, U, V, X, Y, std = data
            pc = ax.pcolormesh(
                X, Y, std, alpha=0.5, shading="auto", cmap="Blues", linewidth=0, antialiased=True, rasterized=True
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
            std = data_dict["var"] # todo
            return x_train, U, V, X, Y, std
        return None

    def data(self, la: Laplace, x: torch.Tensor, N: int):
        # Todo: Think about how to properly calculate the std(norm(VF)), how big are the correlations are there any also see this post https://stats.stackexchange.com/questions/291562/mean-and-variance-of-norm-of-normal-random-variables
        x_train = x
        X = np.linspace(-6, 6, N)
        Y = np.linspace(-6, 6, N)
        U, V = np.meshgrid(X, Y)
        old_model = la.model
        la.model = ODEWrapper(la.model.model.odefunc)
        la.backend.model = la.model
        shape_u = U.shape
        U = torch.tensor(
            U, dtype=torch.float32, device=train_device.get_device()
        ).flatten()
        V = torch.tensor(
            V, dtype=torch.float32, device=train_device.get_device()
        ).flatten()
        data = torch.stack((U, V)).transpose(0, 1)

        # mu_list = list()
        # var_list = list()
        x0 = data.requires_grad_(True)
        f_mu, f_var = la(x0)
        f_std = torch.sqrt(f_var)
        norm = torch.norm(torch.stack((f_std[:, 0, 0], f_std[:, 1, 1]), dim=1), dim=-1)
        std = norm.cpu().numpy().reshape(shape_u)
        U = f_mu[:, 0].detach().cpu().numpy().reshape(shape_u)
        V = f_mu[:, 1].detach().cpu().numpy().reshape(shape_u)
        torch.save(
            {"U": U, "V": V, "X": X, "Y": Y, "std": std, "x_train": x_train},
            os.path.join(self.fig_data, f"{self.NAME}.pt"),
        )
        la.model = old_model
        la.backend.model = old_model


class VectorFieldNormPlotUncertainty(VectorFieldPlotUncertainty):
    NAME = "vector_field_norm_uncertainty"

    def __init__(self, results_dir):
        super(VectorFieldNormPlotUncertainty, self).__init__(results_dir=results_dir)

    def plot_fig(self, ax, fig):
        self.NAME = "vector_field_uncertainty"
        data = self.load()
        self.NAME = "vector_field_norm_uncertainty"
        if data is not None:
            # Todo implement this correctly
            x_train, U, V, X, Y, std = data
            mean = np.linalg.norm(np.stack((U, V), axis=-1), axis=-1)
            std = np.sqrt(std)  # todo
            np_max = np.max(std)
            pc = ax.pcolormesh(
                X, Y, std/mean, alpha=0.5, shading="auto", cmap="Blues", vmin=-0.0, vmax=0.05, linewidth=0,
            )
            pc.set_edgecolor('face')
            for i in range(x_train.shape[0]):
                ax.scatter(x_train[i, :, 0], x_train[i, :, 1], s=0.5, color="k")
            print(np.min(std))
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
        self.NAME = "vector_field_uncertainty"
        data = self.load()
        self.NAME = "vector_field_norm_uncertainty"
        if data is not None:
            x_train, U, V, X, Y, std = data
            mean = np.linalg.norm(np.stack((U, V), axis=-1), axis=-1)
            pc = ax.pcolormesh(
                X, Y, std/mean, alpha=0.5, shading="auto", cmap="Blues", linewidth=0, antialiased=True, rasterized=True
            )
            pc.set_edgecolor('face')
            for i in range(x_train.shape[0]):
                ax.scatter(x_train[i, :, 0], x_train[i, :, 1], s=0.5, color="k")
            ax.quiver(X[::5], Y[::5], U[::5, ::5], V[::5, ::5])
            ax.set_xlabel("$q$")
            ax.set_ylabel("$p$")
            return pc
