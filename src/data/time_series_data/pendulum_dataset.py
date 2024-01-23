import torch

from data.time_series_data.toy_timeseriers_base import TimeSeriesToyBase, ODE, DampedODE
from time_series.models.odeint_options import OdeintOptions


class Pendulum(TimeSeriesToyBase):
    BATCH_SIZE = 16
    T_MAX_TRAIN = 1.5
    T_MAX_TEST = 10
    N_SAMPLES_TRAIN = 100
    N_SAMPLES_TEST = 250

    def __init__(self):
        super(Pendulum, self).__init__()
        self.dataset_name = "pendulum"
        self.has_hamiltonian = True
        self.ode_class = PendulumODE(batch_size=self.BATCH_SIZE)


class PendulumFull(TimeSeriesToyBase):
    BATCH_SIZE = 16
    T_MAX_TRAIN = 3.0
    T_MAX_TEST = 10
    N_SAMPLES_TRAIN = 100
    N_SAMPLES_TEST = 250

    def __init__(self):
        super(PendulumFull, self).__init__()
        self.dataset_name = "pendulum_full"
        self.has_hamiltonian = True
        self.ode_class = PendulumODE(batch_size=self.BATCH_SIZE)


class DampedPendulum(Pendulum):
    BATCH_SIZE = 16
    T_MAX_TRAIN = 3.5
    T_MAX_TEST = 10
    N_SAMPLES_TRAIN = 100
    N_SAMPLES_TEST = 250

    def __init__(self):
        super(DampedPendulum, self).__init__()
        self.dataset_name = "damped_pendulum"
        self.has_hamiltonian = True
        self.ode_class = DampedODE(PendulumODE(batch_size=self.BATCH_SIZE))


class PendulumODE(ODE):
    def __init__(self, batch_size: int):
        super(PendulumODE, self).__init__()

        self.a = -5.0

        x0_1 = torch.normal(1.0, 0.1, size=(batch_size,))
        x0_2 = torch.normal(0.0, 0.1, size=(batch_size,))
        self.x0 = torch.stack((x0_1, x0_2), dim=-1)
        self.opts = OdeintOptions()
        self.opts.solver = "rk4"
        self.opts.step_size = 0.001

    def forward(self, t, x):
        """
        Describes the motion of a simple pendulum without friction
        :param t: time
        :param x: dim 0 - angular displacement, dim 1 angluar velocity
        :param a: parameter dependent on the length of the pendulum
        :return: dx0/dt, dx1/dt
        """
        q, p = torch.chunk(x, chunks=2, dim=-1)
        x1 = x[:, 1]
        x2 = self.a * torch.sin(x[:, 0])
        return torch.stack((x1, x2), dim=-1)

    def hamiltonian(self, x):
        q, p = torch.chunk(x, chunks=2, dim=-1)
        h = p ** 2 / 2 + torch.cos(q) * self.a
        return h


if __name__ == "__main__":
    pendulum = Pendulum()
    pendulum.generate_and_save_data()
    pendulum = PendulumFull()
    pendulum.generate_and_save_data()
