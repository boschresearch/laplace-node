import os
import matplotlib as mpl
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

from plot import init_figure_third, init_figure_half_column


class PlottingClass(ABC):
    def __init__(self, results_dir: str):
        self.fig_data = self.get_fig_data_dir(results_dir)
        self.fig_dir = self.get_fig_dir(results_dir)

    @property
    @classmethod
    @abstractmethod
    def NAME(cls):
        ...

    @staticmethod
    def get_fig_data_dir(results_dir: str):
        fig_data = os.path.join(results_dir, "plot_data")
        if not os.path.exists(fig_data):
            os.makedirs(fig_data)
        return fig_data

    @staticmethod
    def get_fig_dir(results_dir: str):
        fig_dir = os.path.join(results_dir, "plots")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        return fig_dir

    @abstractmethod
    def plot(self, ax: plt.Axes):
        ...

    def plot_no_tick_marks(self, ax: plt.Axes):
        self.plot(ax)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    @abstractmethod
    def load(self):
        ...

    @abstractmethod
    def data(self, *args, **kwargs):
        ...

    def figure(self, x_lim=None, y_lim=None):
        fig, ax = init_figure_half_column()
        self.plot(ax)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        self.save_figure(fig=fig)

    def figure_half_column(self):
        fig, ax = init_figure_half_column()
        pos = mpl.transforms.Bbox([[0.25, 0.3], [0.9, 0.95]])
        ax.set_position(pos)
        self.plot(ax)
        self.save_figure(fig=fig)

    def figure_half_column_no_ticks(self):
        fig, ax = init_figure_half_column()
        pos = mpl.transforms.Bbox([[0.1, 0.2], [0.9, 0.95]])
        ax.set_position(pos)
        self.plot_no_tick_marks(ax)
        self.save_figure(fig=fig, name=f"{self.NAME}_no_ticks")

    def save_figure(self, fig, name=None):
        if name is None:
            name = self.NAME
        fig.savefig(os.path.join(self.fig_dir, f"{name}.png"), dpi=500)
        plt.clf()
        plt.close(fig=fig)
