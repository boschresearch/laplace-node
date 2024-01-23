import os

import numpy as np
import torch

from time_series.plotting.plotting_class import PlottingClass
from util.device_setter import train_device


class VectorFieldPlot(PlottingClass):
    NAME = "vector_field"

    def __init__(self, results_dir):
        super(VectorFieldPlot, self).__init__(results_dir=results_dir)

    def plot(self, ax):
        data = self.load()
        if data is not None:
            U, V, X, Y, x_train = data
            for i in range(x_train.size(0)):
                ax.scatter(x_train[i, :, 0], x_train[i, :, 1], s=0.5, color="k")
            ax.quiver(X, Y, U, V)
            ax.set_xlabel("$q$")
            ax.set_ylabel("$p$")

    def load(self):
        file = os.path.join(self.fig_data, f"{self.NAME}.pt")
        if os.path.exists(file):
            data_dict = torch.load(os.path.join(self.fig_data, f"{self.NAME}.pt"))
            U = data_dict["U"]
            V = data_dict["V"]
            X = data_dict["X"]
            Y = data_dict["Y"]
            x_train = data_dict["x_train"]
            return U, V, X, Y, x_train
        else:
            return None

    def data(self, N, model, x):
        x1_range = 6
        x2_range = 8
        X = np.linspace(-x1_range, x1_range, N)
        Y = np.linspace(-x2_range, x2_range, N)
        U, V = np.meshgrid(X, Y)
        shape_u = U.shape
        U = torch.tensor(U, dtype=torch.float32, device=train_device.get_device()).flatten()
        V = torch.tensor(V, dtype=torch.float32, device=train_device.get_device()).flatten()
        data = torch.stack((U, V)).transpose(0, 1)
        vec = None
        for i in range(data.size()[0]):
            if vec is None:
                vec = model.odefunc(None, data[i, :].unsqueeze(0).requires_grad_(True))
            else:
                vec = torch.cat(
                    (
                        vec,
                        model.odefunc(
                            None, data[i, :].unsqueeze(0).requires_grad_(True)
                        ),
                    ),
                    dim=0,
                )
        U = vec[:, 0].detach().cpu().numpy().reshape(shape_u)
        V = vec[:, 1].detach().cpu().numpy().reshape(shape_u)
        torch.save(
            {"U": U, "V": V, "X": X, "Y": Y, "x_train": x},
            os.path.join(self.fig_data, "vector_field.pt"),
        )
