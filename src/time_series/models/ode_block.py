import torch
from torch import nn as nn
from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint

from options.options_enum import AutodifEnum
from time_series.models.odeint_options import OdeintOptions


class ODEBlock(nn.Module):
    def __init__(self, odefunc, opts: OdeintOptions):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.atol = opts.atol
        self.rtol = opts.rtol
        self.solver = opts.solver
        self.autodif = opts.autodif
        self.step_size = opts.step_size

        self.integration_time = torch.tensor([0, 1], dtype=torch.float64)
        self.options = self.init_options()

    def init_options(self):
        if self.step_size is not None:
            options = {"step_size": self.step_size}
        else:
            options = {}

        return options

    @property
    def autodif(self):
        return self._autodif

    @autodif.setter
    def autodif(self, value):
        self._autodif = value
        self._set_ode_solver()

    def _set_ode_solver(self):
        if self._autodif == AutodifEnum.adjoint:
            self.ode_solver = odeint_adjoint
        elif self._autodif == AutodifEnum.naive:
            self.ode_solver = odeint
        else:
            raise NotImplementedError(
                f"The backpropagation method with name {self._autodif} is not available. You"
                f"can chose form the following list f{AutodifEnum.list()}"
            )

    def forward(self, x: torch.Tensor, t: torch.Tensor = None,) -> torch.Tensor:
        """
        :param x: Initial value for the ODE
        :param t: Time interval on which the ODE is supposed to be evaluated
        :return: solution shape Steps x Batch size x dime1 x dim2....
        """
        if t is not None:
            self.integration_time = t

        integration_time = self.integration_time.type_as(x).to(x)
        result = self.ode_solver(
            self.odefunc,
            x,
            integration_time,
            rtol=self.rtol,
            atol=self.atol,
            method=self.solver,
            options=self.options,
        ).transpose(0, 1)

        return result
