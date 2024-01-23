import torch

from torchdiffeq._impl.odeint import SOLVERS
from torchdiffeq._impl.solvers import FixedGridODESolver


class SymplecticEuler(FixedGridODESolver):
    order = 1

    def _step_func(self, func, t0, dt, t1, y0):
        q, p = torch.chunk(y0, 2, -1)
        f0 = func(t0, y0)
        dp = torch.chunk(f0, 2, -1)[-1] * dt
        p = p + dp
        x_new = torch.cat((q, p), dim=-1)
        dq = torch.chunk(func(t0, x_new), 2, -1)[0] * dt
        dx = torch.cat((dq, dp), dim=-1)
        return dx, f0


def add_symplectic_euler():
    """
    Monkey patches the SOLVER dict in torchdiffeq and adds a new solver
    """
    SOLVERS["symplectic_euler"] = SymplecticEuler
