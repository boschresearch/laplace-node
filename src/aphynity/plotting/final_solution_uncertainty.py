import os

import numpy as np
import torch
from laplace import Laplace

from time_series.plotting.plot_final_solution_uncertainty import (
    FinalSolutionUncertaintyPlot,
)
from util.device_setter import train_device

"""
The following code is adapted from APHYNITY Commit 1b0fcfc7
( https://github.com/yuan-yin/APHYNITY
Copyright (c) 2021 Yuan Yin, Vincent Le Guen, Jérémie Dona, licensed under the MIT License,
cf. 3rd-party-license.txt file in the root directory of this source tree)
"""


class AphinityFinalSolutionUncertaintyPlot(FinalSolutionUncertaintyPlot):

    NAME = "final_solution_uncertainty"

    def data(self, la: Laplace, train_data, x0, t, x, index: int = 0):
        t_max_test = torch.max(t).item() * 4
        t_test = torch.linspace(0, t_max_test, 200)
        true_solution = train_data.true_solution(
            t_test.detach().cpu().numpy(), x0[index].detach().cpu().numpy()
        ).T
        la.model.t = t_test
        x0 = x0.to(train_device.get_device())
        f_mu, f_var = la(x0[index].unsqueeze(0))
        t_test = t_test.flatten().cpu().numpy()
        f_mu = f_mu.squeeze().detach().cpu().numpy().reshape(2, t_test.shape[0])
        f_sigma = torch.diagonal(f_var.squeeze().sqrt().cpu()).numpy()
        pred_std = np.sqrt(f_sigma ** 2 + la.sigma_noise.item() ** 2).reshape(
            2, t_test.shape[0]
        )
        if x is not None:
            x = x[index].transpose(0, 1)
        torch.save(
            {
                "t_train": t,
                "t_test": t_test,
                "x_train": x,
                "f_mu": f_mu.T,
                "pred_std": pred_std.T,
                "true_solution": true_solution,
            },
            os.path.join(self.fig_data, f"{self.NAME}.pt"),
        )
