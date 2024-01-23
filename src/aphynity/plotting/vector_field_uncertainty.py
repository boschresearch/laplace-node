import os

import numpy as np
import torch
from laplace import Laplace
from torch import nn

from time_series.plotting.plot_vector_field_uncertainty import (
    VectorFieldPlotUncertainty,
)
from util.device_setter import train_device

"""
The following code is adapted from APHYNITY Commit 1b0fcfc7
( https://github.com/yuan-yin/APHYNITY
Copyright (c) 2021 Yuan Yin, Vincent Le Guen, Jérémie Dona, licensed under the MIT License,
cf. 3rd-party-license.txt file in the root directory of this source tree)
"""


class AphinityVectorFieldPlotUncertainty(VectorFieldPlotUncertainty):
    def data(self, la: Laplace, x: torch.Tensor, N: int):
        x_train = x
        X = np.linspace(-7, 15, N)
        Y = np.linspace(-2, 3, N)
        U, V = np.meshgrid(X, Y)
        old_model = la.model
        la.model = ODEWrapper(la.model.model.derivative_estimator)
        shape_u = U.shape
        U = torch.tensor(
            U, dtype=torch.float32, device=train_device.get_device()
        ).flatten()
        V = torch.tensor(
            V, dtype=torch.float32, device=train_device.get_device()
        ).flatten()
        data = torch.stack((U, V)).transpose(0, 1)

        mu_list = list()
        var_list = list()
        for i in range(data.size()[0]):
            x0 = data[i, :].unsqueeze(0).requires_grad_(True)
            f_mu, f_var = la(x0)
            mu_list.append(f_mu)
            var_list.append(torch.diagonal(f_var.squeeze()))
        stack = torch.stack(var_list)
        norm = torch.norm(stack, dim=-1)
        var = norm.cpu().numpy().reshape(shape_u)
        vec = torch.stack(mu_list)
        U = vec[:, 0, 0].detach().cpu().numpy().reshape(shape_u)
        V = vec[:, 0, 1].detach().cpu().numpy().reshape(shape_u)
        torch.save(
            {
                "U": U,
                "V": V,
                "X": X,
                "Y": Y,
                "var": var,
                "x_train": x_train.transpose(1, 2),
            },
            os.path.join(self.fig_data, f"{self.NAME}.pt"),
        )
        la.model = old_model


class ODEWrapper(nn.Module):
    def __init__(self, model):
        super(ODEWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(None, x)
