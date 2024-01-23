import torch
from time_series.models.ode_block import ODEBlock

from data.time_series_data.toy_timeseriers_base import TimeSeriesToyBase, ODE
from time_series.models.odeint_options import OdeintOptions


class LotkaVolterraPoly(TimeSeriesToyBase):
    BATCH_SIZE = 1
    T_MAX_TRAIN = 10.0
    T_MAX_TEST = 20.0
    N_SAMPLES_TRAIN = 200
    N_SAMPLES_TEST = 200
    DATA_NOISE = 0.00

    def __init__(self):
        super(LotkaVolterraPoly, self).__init__()
        self.dataset_name = "lotka_volterra_poly"
        self.dim = 2
        self.has_hamiltonian = False
        self.ode_class = LotkaVolterraODE(batch_size=self.BATCH_SIZE)

    def get_dataset(self, split):

        self.generate_and_save_data()

        if split == "train":
            dataset = self.load_data_train()
        else:
            dataset = self.load_data_test()
        return dataset

    def dataset_generation(
            self, N: int = 1000, t_max: float = 50
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        ode = self.ode_class
        t = torch.linspace(0, t_max, N)
        model = ODEBlock(ode, opts=ode.opts)
        out = model(ode.x0, t).detach()
        noise = torch.normal(torch.zeros(out.shape), self.DATA_NOISE)
        return t, (out + noise), ode.x0



class LotkaVolterraODE(ODE):
    def __init__(self, batch_size: int):
        super(LotkaVolterraODE, self).__init__()
        """Parameters for Lotka Volterra DQ"""
        self.a = 1.5
        self.b = 1.
        self.c = 3.0
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
    lv = TrigonometricData()
    lv.generate_and_save_data()
