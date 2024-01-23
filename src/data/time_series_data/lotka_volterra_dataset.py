import torch

from data.time_series_data.toy_timeseriers_base import TimeSeriesToyBase, ODE
from time_series.models.odeint_options import OdeintOptions


class LotkaVolterraFull(TimeSeriesToyBase):
    BATCH_SIZE = 1
    T_MAX_TRAIN = 10.0
    T_MAX_TEST = 20.0
    N_SAMPLES_TRAIN = 50
    N_SAMPLES_TEST = 250
    DATA_NOISE = 0.03

    def __init__(self):
        super(LotkaVolterraFull, self).__init__()
        self.dataset_name = "lotka_volterra_full"
        self.dim = 2
        self.has_hamiltonian = False
        self.ode_class = LotkaVolterraODE(batch_size=self.BATCH_SIZE)


class LotkaVolterraHalf(TimeSeriesToyBase):
    BATCH_SIZE = 1
    T_MAX_TRAIN = 5.0
    T_MAX_TEST = 20.0
    N_SAMPLES_TRAIN = 50
    N_SAMPLES_TEST = 250
    DATA_NOISE = 0.03

    def __init__(self):
        super(LotkaVolterraHalf, self).__init__()
        self.dataset_name = "lotka_volterra_half"
        self.dim = 2
        self.has_hamiltonian = False
        self.ode_class = LotkaVolterraODE(batch_size=self.BATCH_SIZE)


class LotkaVolterraODE(ODE):
    def __init__(self, batch_size: int):
        super(LotkaVolterraODE, self).__init__()
        """Parameters for Lotka Volterra DQ"""
        self.a = 2 / 3
        self.b = 4 / 3
        self.c = 1.0
        self.d = 1.0

        self.x0 = torch.ones((batch_size, 2))
        self.opts = OdeintOptions()
        self.opts.solver = "dopri5"
        self.opts.rtol = 1e-5
        self.opts.rtol = 1e-5

    def forward(self, t, x: torch.Tensor):
        """
        Describes the Lotka Volterra DEQ
        :param t: time
        :param x: number of prey and predator
        :param a: positive parameter
        :param b: positive parameter
        :param c: positive parameter
        :param d: positive parameter
        :return: dx/dt instantaneous growth rates
        """
        x1 = self.a * x[:, 0] - self.b * x[:, 0] * x[:, 1]
        x2 = -self.d * x[:, 1] + self.c * x[:, 0] * x[:, 1]
        return torch.stack((x1, x2), dim=-1)


if __name__ == "__main__":
    lv = LotkaVolterraFull()
    lv.generate_and_save_data()
    lv = LotkaVolterraHalf()
    lv.generate_and_save_data()
