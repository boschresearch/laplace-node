import torch
from time_series.models.ode_block import ODEBlock

from data.time_series_data.toy_timeseriers_base import TimeSeriesToyBase, ODE
from time_series.models.odeint_options import OdeintOptions


class TrigonometricData(TimeSeriesToyBase):
    BATCH_SIZE = 1
    T_MAX_TRAIN = 40.0
    T_MAX_TEST = 80.0
    N_SAMPLES_TRAIN = 200
    N_SAMPLES_TEST = 250
    DATA_NOISE = 0.0

    def __init__(self):
        super(TrigonometricData, self).__init__()
        self.dataset_name = "trigonometric_ode"
        self.dim = 2
        self.has_hamiltonian = False
        self.ode_class = TrigonometricODE(batch_size=self.BATCH_SIZE)

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


class TrigonometricODE(ODE):
    def __init__(self, batch_size: int):
        super(TrigonometricODE, self).__init__()

        self.x0 = torch.tensor([[0.5, 1.]])
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
        x1 = torch.cos(x[:, 1])
        x2 = -torch.sin(x[:, 0])
        return torch.stack((x1, x2), dim=-1)


if __name__ == "__main__":
    lv = TrigonometricData()
    lv.generate_and_save_data()
