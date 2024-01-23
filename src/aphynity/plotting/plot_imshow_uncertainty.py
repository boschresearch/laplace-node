import os

import numpy as np
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


class ImshowUncertaintyPlot(PlottingClass):
    NAME = "imshow_uncertainty"
    num_images = 5

    def figure(self, lim_solution, lim_uncertainty, lim_calibration):
        images_ = int(self.num_images + 1)
        fig, ax = plt.subplots(4, images_)
        fig.set_size_inches(3.25, 1.8)
        self.plot(ax, fig, lim_solution, lim_uncertainty, lim_calibration)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        self.save_figure(fig=fig)

    def plot(
        self,
        axs: plt.Axes,
        fig,
        lim_solution,
        lim_uncertainty,
        lim_calibration,
        dim=0,
        size=5,
    ):
        result, std, true_solution = self.load()
        shape_ = result.shape[2]

        ax = axs[0, 0]
        ax.axis("off")
        ax.text(0, 0.5, "True solution", size=size)
        ax = axs[1, 0]
        ax.axis("off")
        ax.text(0, 0.5, "Augmented\nmodel", size=size)
        ax = axs[2, 0]
        ax.axis("off")
        ax.text(0, 0.5, "Uncertainty", size=size)
        ax = axs[3, 0]
        ax.axis("off")
        ax.text(0, 0.5, "$\gamma$", size=size)

        for k in range(0, shape_):
            i = k + 1
            ax = axs[0, i]
            j = k
            im = ax.imshow(
                true_solution[j, dim], extent=[0, 1, 0, 1], interpolation="none"
            )
            cmap_max = 0.4
            im.set_clim(lim_solution)
            if k < self.num_images / 2:
                for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
                    ax.spines[side].set_visible(True)
                    ax.spines[side].set_linewidth(1)
                    ax.spines[side].set_edgecolor("orange")
                    ax.set_xticks([])
                    ax.set_yticks([])
            else:
                ax.axis("off")
            if k == shape_ - 1:
                self.plot_colorbar(ax, fig, im, size - 1)
            ax = axs[1, i]
            im = ax.imshow(result[0, dim, j], interpolation="none")
            im.set_clim(lim_solution)
            ax.axis("off")
            if k == shape_ - 1:
                self.plot_colorbar(ax, fig, im, size - 1)
            ax = axs[2, i]
            im = ax.imshow(std[0, dim, j], interpolation="none")
            l = [0.018672453, 0.018705264]
            im.set_clim(lim_uncertainty)
            if k == shape_ - 1:
                self.plot_colorbar(ax, fig, im, size - 1)
            print(np.max(std[0, dim, j]))
            print(np.min(std[0, dim, j]))
            ax.axis("off")
            ax = axs[3, i]
            err = np.abs(true_solution[j, dim] - result[0, dim, j]) / std[0, dim, j]
            print("==========================")
            print(np.min(err))
            print(np.max(err))
            print("==========================")
            im = ax.imshow(err, interpolation="none")
            l1 = [0.0, 0.6]
            im.set_clim(lim_calibration)
            ax.axis("off")
            if k == shape_ - 1:
                self.plot_colorbar(ax, fig, im, size - 1)

    def plot_colorbar(self, ax, fig, im, size):
        cbar = fig.colorbar(
            im, ax=ax, shrink=1.0, location="right", orientation="vertical"
        )
        cbar.ax.tick_params(labelsize=size)
        cbar.ax.ticklabel_format(useOffset=False)

    def load(self):
        data_dict = torch.load(os.path.join(self.fig_data, f"{self.NAME}.pt"))
        f = data_dict["f"]
        std = data_dict["std"]
        true_solution = data_dict["true_solution"]
        return f, std, true_solution

    def data(self, la, train_data, x0, t, index: int = 0):
        interval = 1
        im_size = x0.shape[-1]
        new_im_size = int(im_size / interval)
        t_max_test = torch.max(t).item() * 2
        t_test = torch.linspace(0, t_max_test, self.num_images)
        la.model.model.si = interval
        la.model.t = t_test
        f_mu, f_var = la(x0[index].unsqueeze(0).to(train_device.get_device()))
        f_mu = (
            f_mu.squeeze()
            .detach()
            .cpu()
            .numpy()
            .reshape(1, 2, self.num_images, new_im_size, new_im_size)
        )
        f_sigma = torch.diagonal(f_var.detach().cpu().squeeze().sqrt()).numpy()
        pred_std = np.sqrt(f_sigma ** 2 + la.sigma_noise.item() ** 2).reshape(
            1, 2, self.num_images, new_im_size, new_im_size
        )
        true_solution = train_data.true_solution(t_test, x0[index])
        torch.save(
            {
                "f": f_mu,
                "std": pred_std,
                "true_solution": true_solution.detach().cpu().numpy(),
            },
            os.path.join(self.fig_data, f"{self.NAME}.pt"),
        )
