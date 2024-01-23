import os

import torch

from time_series.plotting.plot_final_solution import FinalSolutionPlot
from util.device_setter import train_device

"""
The following code is adapted from APHYNITY Commit 1b0fcfc7
( https://github.com/yuan-yin/APHYNITY
Copyright (c) 2021 Yuan Yin, Vincent Le Guen, Jérémie Dona, licensed under the MIT License,
cf. 3rd-party-license.txt file in the root directory of this source tree)
"""


class AphinityFinalSolutionPlot(FinalSolutionPlot):
    NAME = "final_solution"

    def data(self, train_data, t_train, t_test, x0, x_train, x_test, model, index: int = 0):
        t_max_test = torch.max(t_train).item() * 5
        t_test = torch.linspace(0, t_max_test, 200, device=train_device.get_device())
        x0 = x0.requires_grad_(True).to(train_device.get_device())
        result = model(x0, t_test).detach().cpu()[index].transpose(0, 1)
        true_solution = train_data.true_solution(t_test.detach().cpu().numpy(), x0[index].detach().cpu().numpy()).T
        x_train = x_train[index].transpose(0, 1)
        t_max_train = torch.max(t_train)
        torch.save(
            {
                "t_test": t_test.detach().cpu().numpy(),
                "t_train": t_train,
                "t_max_train": t_max_train,
                "x_train": x_train,
                "f": result,
                "true_solution": true_solution
            },
            os.path.join(self.fig_data, f"{self.NAME}.pt"),
        )
