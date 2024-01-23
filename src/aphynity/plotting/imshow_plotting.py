import os

import torch
from matplotlib import pyplot as plt

from time_series.plotting.plotting_class import PlottingClass
from util.device_setter import train_device

"""
The following code is adapted from APHYNITY Commit 1b0fcfc7
( https://github.com/yuan-yin/APHYNITY
Copyright (c) 2021 Yuan Yin, Vincent Le Guen, Jérémie Dona, licensed under the MIT License,
cf. 3rd-party-license.txt file in the root directory of this source tree)
"""


class ImshowPlot(PlottingClass):
    NAME = "imshow"
    num_images = 10

    def figure(self, x_lim=None, y_lim=None):
        fig, ax = plt.subplots(2, 10)
        fig.set_size_inches(3, 1.)
        self.plot(ax)
        # fig.tight_layout()
        self.save_figure(fig=fig)

    def plot(self, axs: plt.Axes):
        result, true_solution = self.load()
        print("plottingimshow")
        for i in range(0, result.shape[2]):
            ax = axs[0, i]
            im = ax.imshow(result[0, 1, i])
            cmap_max = 0.2
            # im.set_clim(0., cmap_max)
            ax.axis("off")
            ax = axs[1, i]
            im = ax.imshow(true_solution[i, 1], extent=[0, 1, 0, 1])
            # im.set_clim(0., cmap_max)
            if i < 5:
                for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
                    ax.spines[side].set_visible(True)
                    ax.spines[side].set_linewidth(1)
                    ax.spines[side].set_edgecolor("red")
                    ax.set_xticks([])
                    ax.set_yticks([])
            else:
                ax.axis("off")

    def load(self):
        data_dict = torch.load(os.path.join(self.fig_data, f"{self.NAME}.pt"))
        f = data_dict["f"]
        true_solution = data_dict["true_solution"]
        return f, true_solution

    def data(self, model, train_data, x0, t):
        index = 160
        t_max_test = torch.max(t).item() * 2
        t_test = torch.linspace(0, t_max_test, self.num_images)
        result = model(x0[index].unsqueeze(0).to(train_device.get_device()), t_test)
        true_solution = train_data.true_solution(t_test, x0[index])
        torch.save(
            {
                "f": result.detach().cpu().numpy(),
                "true_solution": true_solution.detach().cpu().numpy()
            },
            os.path.join(self.fig_data, f"{self.NAME}.pt")
    )
