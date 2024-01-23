import math
import shelve
from collections import OrderedDict

import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch.utils.data import Dataset

MAX = np.iinfo(np.int32).max

"""
The following code is adapted from APHYNITY Commit 1b0fcfc7
( https://github.com/yuan-yin/APHYNITY
Copyright (c) 2021 Yuan Yin, Vincent Le Guen, Jérémie Dona, licensed under the MIT License,
cf. 3rd-party-license.txt file in the root directory of this source tree)
"""


class DampledPendulum(Dataset):
    __default_params = OrderedDict(omega0_square=(2 * math.pi / 12) ** 2, alpha=0.1)

    def __init__(self, path, num_seq, time_horizon, dt, params=None, group="train"):
        super().__init__()
        self.len = num_seq
        self.time_horizon = float(time_horizon)  # total time
        self.dt = float(dt)  # time step
        self.t_eval = np.arange(0, self.time_horizon, self.dt)
        self.params = OrderedDict()
        if params is None:
            self.params.update(self.__default_params)
        else:
            self.params.update(params)
        self.group = group
        self.data = shelve.open(path)
        self.jitter = 0.1

    def _f(self, t, x):  # coords = [q,p]
        omega0_square, alpha = list(self.params.values())

        q, p = np.split(x, 2)
        dqdt = p
        dpdt = -omega0_square * np.sin(q) - alpha * p
        return np.concatenate([dqdt, dpdt], axis=-1)

    def _get_initial_condition(self, seed):
        np.random.seed(seed if self.group == "train" else MAX - seed)
        y0 = np.random.rand(2) * 1.0
        radius = np.random.rand() + 1.3
        y0 = y0 / np.sqrt((y0 ** 2).sum()) * radius
        return y0

    def __getitem__(self, index: int):
        t_eval = np.arange(0, self.time_horizon, self.dt)
        if self.data.get(str(index)) is None:
            y0 = self._get_initial_condition(index)
            states = self.true_solution(t_eval, y0)
            noise = np.random.randn(*np.shape(states))*self.jitter
            noise[:, 0] = noise[:, 0] * 0
            states = states + noise
            self.data[str(index)] = states
            states = torch.from_numpy(states).float()
        else:
            data_torch = torch.from_numpy(self.data[str(index)])
            states = data_torch.float()
        t_eval_torch = torch.from_numpy(t_eval)
        return {"states": states, "t": t_eval_torch.float()}

    def true_solution(self, t_eval, y0):
        states = solve_ivp(
            fun=self._f,
            t_span=(0, np.amax(t_eval)),
            y0=y0,
            method="DOP853",
            t_eval=t_eval,
            rtol=1e-10,
        ).y
        return states

    def __len__(self):
        return self.len
