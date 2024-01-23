from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from util.device_setter import train_device

"""
The following code is adapted from APHYNITY Commit 1b0fcfc7
( https://github.com/yuan-yin/APHYNITY
Copyright (c) 2021 Yuan Yin, Vincent Le Guen, Jérémie Dona, licensed under the MIT License,
cf. 3rd-party-license.txt file in the root directory of this source tree)
"""


class ReactionDiffusionParamPDE(nn.Module):
    def __init__(self, dx, is_complete, real_params=None):
        super().__init__()

        self._dx = dx
        self._laplacian = nn.Parameter(
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0],],).float().view(1, 1, 3, 3)
            / (self._dx * self._dx),
            requires_grad=False,
        )

        self.real_params = real_params
        self.is_complete = is_complete
        self.params_org = nn.ParameterDict(
            {
                "a_org": nn.Parameter(torch.tensor(-2.0)),
                "b_org": nn.Parameter(torch.tensor(-2.0)),
                "k_org": nn.Parameter(torch.tensor(-2.0)),
            }
        )

        self.params = OrderedDict()
        if self.real_params is not None:
            self.params.update(real_params)

    def forward(self, state):
        U = state[:, :1]
        V = state[:, 1:]

        if self.real_params is None:
            self.params["a"] = torch.sigmoid(self.params_org["a_org"]) * 1e-2
            self.params["b"] = torch.sigmoid(self.params_org["b_org"]) * 1e-2

        U_ = F.pad(U, pad=(1, 1, 1, 1), mode="circular")
        Delta_u = F.conv2d(U_, self._laplacian)

        V_ = F.pad(V, pad=(1, 1, 1, 1), mode="circular")
        Delta_v = F.conv2d(V_, self._laplacian)

        if self.is_complete:
            if self.real_params is None:
                self.params["k"] = torch.sigmoid(self.params_org["k_org"]) * 1e-2
            (a, b, k) = list(self.params.values())
            dUdt = a * Delta_u + U - U.pow(3) - V - k
            dVdt = b * Delta_v + U - V
        else:
            (a, b) = list(self.params.values())
            dUdt = a * Delta_u
            dVdt = b * Delta_v

        return torch.cat([dUdt, dVdt], dim=1)


class DampedWaveParamPDE(nn.Module):
    def __init__(self, is_complete, real_params=None):
        super().__init__()
        self.real_params = real_params
        self.is_complete = is_complete
        self.params_org = nn.ParameterDict(
            {
                "k_org": nn.Parameter(torch.tensor(45.0)),
                "c_org": nn.Parameter(torch.tensor(3.6)),
            }
        )
        self.params = OrderedDict()
        if self.real_params is not None:
            self.params.update(real_params)

        self._sobel_ddx = (
            torch.tensor(
                [[0, 0, 0], [1, -2, 1], [0, 0, 0]], device=train_device.get_device()
            )
            .float()
            .view(1, 1, 3, 3)
        )
        self._sobel_ddy = (
            torch.tensor(
                [[0, 1, 0], [0, -2, 0], [0, 1, 0]], device=train_device.get_device()
            )
            .float()
            .view(1, 1, 3, 3)
        )

    def forward(self, state):
        if self.real_params is None:
            self.params["c"] = self.params_org["c_org"] * 100

        wave = state[:, 0:1].clone()

        ddx = F.conv2d(wave, self._sobel_ddx)
        ddy = F.conv2d(wave, self._sobel_ddy)

        lap = F.pad(ddx + ddy, (1, 1, 1, 1))
        wave_diff = state[:, 1:2]

        if self.is_complete:
            if self.real_params is None:
                self.params["k"] = self.params_org["k_org"]
            (c, k) = list(self.params.values())
            wave_for_calculus = state[:, 1:2].clone()
            corrected_laplacian = c ** 2 * lap - (k * wave_for_calculus)
        else:
            (c,) = list(self.params.values())
            corrected_laplacian = c ** 2 * lap

        return torch.cat([wave_diff, corrected_laplacian], dim=1)


class DampedPendulumParamPDE(nn.Module):
    def __init__(self, is_complete=False, real_params=None):
        super().__init__()
        self.real_params = real_params
        self.is_complete = is_complete
        self.params_org = nn.ParameterDict(
            {
                "omega0_square_org": nn.Parameter(torch.tensor(0.2)),
                "alpha_org": nn.Parameter(torch.tensor(0.1)),
            }
        )
        self.params = OrderedDict()
        if real_params is not None:
            self.params.update(real_params)

    def forward(self, state):
        if self.real_params is None:
            self.params["omega0_square"] = self.params_org["omega0_square_org"]

        q = state[:, 0:1]
        p = state[:, 1:2]

        if self.is_complete:
            if self.real_params is None:
                self.params["alpha"] = self.params_org["alpha_org"]
            (omega0_square, alpha) = list(self.params.values())
            dqdt = p
            dpdt = -omega0_square * torch.sin(q) - alpha * p
        else:
            (omega0_square,) = list(self.params.values())
            dqdt = p
            dpdt = -omega0_square * torch.sin(q)

        return torch.cat([dqdt, dpdt], dim=1)


class ZeroPredictor(nn.Module):
    def __init__(self):
        super(ZeroPredictor, self).__init__()
        self.params = torch.Tensor([0])

    def forward(self, x: torch.Tensor):
        """
        Returns  0, enables using the current architecture to use a plain neural ODE
        Parameters
        ----------
        x

        Returns
        -------

        """
        return 0 * x


class ConvNetEstimator(nn.Module):
    def __init__(self, state_c=2, hidden=16):
        super().__init__()
        kernel_size = 3
        padding = kernel_size // 2
        self.state_c = state_c
        self.net = nn.Sequential(
            nn.Conv2d(
                state_c, hidden, kernel_size=kernel_size, padding=padding, bias=False
            ),
            # nn.BatchNorm2d(hidden, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv2d(
                hidden, hidden, kernel_size=kernel_size, padding=padding, bias=False
            ),
            # nn.BatchNorm2d(hidden, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv2d(hidden, state_c, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x):
        return self.net(x)

    def get_derivatives(self, x):
        batch_size, nc, T, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size * T, nc, h, w)
        x = self.forward(x)
        x = x.view(batch_size, T, self.state_c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x


class MLP(nn.Module):
    def __init__(self, state_c, hidden):
        super().__init__()
        self.state_c = state_c
        self.net = nn.Sequential(
            nn.Linear(state_c, hidden),
            # nn.Tanh(),
            # nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, state_c),
        )

    def forward(self, x):
        return self.net(x)

    def get_derivatives(self, x):
        batch_size, nc, T = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(batch_size * T, nc)
        x = self.forward(x)
        x = x.view(batch_size, T, self.state_c)
        x = x.permute(0, 2, 1).contiguous()
        return x
