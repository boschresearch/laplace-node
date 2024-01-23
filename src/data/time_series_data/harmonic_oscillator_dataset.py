import math

import torch

from data.time_series_data.toy_timeseriers_base import TimeSeriesToyBase, ODE, DampedODE
from time_series.models.ode_block import ODEBlock
from time_series.models.odeint_options import OdeintOptions
from util.device_setter import train_device


class HarmonicOscillator(TimeSeriesToyBase):
    BATCH_SIZE = 16
    SPRING_CONSTANT = 2
    T_MAX_TRAIN = math.pi / math.sqrt(SPRING_CONSTANT)
    T_MAX_TEST = 10
    N_SAMPLES_TRAIN = 50
    N_SAMPLES_TEST = 250
    DATA_NOISE = 0.3

    def __init__(self, dim=1):
        super(HarmonicOscillator, self).__init__()
        self.DIMENSION = 2 * dim
        self.has_hamiltonian = True
        self.dataset_name = f"harmonic_oscillator_{dim}"
        s0 = torch.normal(
            3.0, 0.2, size=(self.BATCH_SIZE, dim), device=train_device.get_device()
        )
        v0 = torch.normal(
            -0.0, 0.2, size=(self.BATCH_SIZE, dim), device=train_device.get_device()
        )
        self.x0 = torch.cat((s0, v0), dim=-1)
        self.ode_class = HarmonicOscillatorODE(
            x0=self.x0,
            batch_size=self.BATCH_SIZE, spring_constant=self.SPRING_CONSTANT
        )


class HarmonicOscillatorHalf(HarmonicOscillator):

    def __init__(self, dim=1):
        super(HarmonicOscillatorHalf, self).__init__()
        self.dataset_name = f"harmonic_oscillator_half"
        s0 = torch.normal(
            .0, 0.2, size=(self.BATCH_SIZE, dim), device=train_device.get_device()
        )
        v_initial = 3. * math.sqrt(self.SPRING_CONSTANT)
        v0 = torch.normal(
            v_initial, 0.2, size=(self.BATCH_SIZE, dim), device=train_device.get_device()
        )
        self.x0 = torch.cat((s0, v0), dim=-1)
        self.ode_class = HarmonicOscillatorODE(
            x0=self.x0,
            batch_size=self.BATCH_SIZE, spring_constant=self.SPRING_CONSTANT
        )


class HarmonicOscillatorFull(HarmonicOscillator):
    SPRING_CONSTANT = 2
    T_MAX_TRAIN = 2 * math.pi / math.sqrt(SPRING_CONSTANT)
    T_MAX_TEST = 8 * math.pi / math.sqrt(SPRING_CONSTANT)

    def __init__(self, dim=1):
        super(HarmonicOscillatorFull, self).__init__(dim)
        self.dataset_name = "harmonic_oscillator_full"


class GappedHarmonicOscillator(HarmonicOscillator):
    def __init__(self):
        super(GappedHarmonicOscillator, self).__init__()
        self.dataset_name = f"harmonic_oscillator_gap"

    def dataset_generation(
        self, N: int = 1000, t_max: float = 50
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        ode = self.ode_class
        n = int(N / 3)
        dt = 2 * math.pi / math.sqrt(2) / 8
        t0 = 0.0
        t1 = dt
        t2 = math.pi / math.sqrt(2) - dt
        t3 = math.pi / math.sqrt(2) + dt
        t4 = 2 * math.pi / math.sqrt(2) - dt
        t5 = 2 * math.pi / math.sqrt(2) + dt
        t = torch.cat(
            (
                torch.linspace(t0, t1, n),
                torch.linspace(t2, t3, n),
                torch.linspace(t4, t5, n),
            ),
            dim=0,
        )
        model = ODEBlock(ode, opts=ode.opts)
        out = model(ode.x0, t)
        noise = torch.normal(torch.zeros(out.shape), self.DATA_NOISE)
        return t, (out + noise), ode.x0


class OffsetHarmonicOscillator(HarmonicOscillator):
    def __init__(self):
        super(OffsetHarmonicOscillator, self).__init__()
        self.dataset_name = f"harmonic_oscillator_offset"

    def dataset_generation(
        self, N: int = 1000, t_max: float = 50
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        ode = self.ode_class
        n = int(N / 3)
        dt = 2 * math.pi / math.sqrt(2) / 3
        t = torch.cat(
            (
                torch.linspace(0.0, dt, n),
                torch.linspace(2 * dt, 3 * dt, n),
                torch.linspace(4 * dt, 5 * dt, n),
            ),
            dim=0,
        )
        model = ODEBlock(ode, opts=ode.opts)
        out = model(ode.x0, t)
        noise = torch.normal(torch.zeros(out.shape), 0.01)
        return t, (out + noise), ode.x0


class HarmonicOscillatorThreeQuarter(HarmonicOscillator):
    BATCH_SIZE = 16
    SPRING_CONSTANT = 2
    T_MAX_TRAIN = 3/2 * math.pi / math.sqrt(SPRING_CONSTANT)
    T_MAX_TEST = 20

    def __init__(self):
        super(HarmonicOscillatorThreeQuarter, self).__init__()
        self.dataset_name = f"harmonic_oscillator_three_quarter"


class DampedHarmonicOscillator(HarmonicOscillator):
    BATCH_SIZE = 16
    SPRING_CONSTANT = 2
    T_MAX_TRAIN = 3/2 * math.pi / math.sqrt(SPRING_CONSTANT)
    T_MAX_TEST = 50

    def __init__(self, dim=1):
        super(DampedHarmonicOscillator, self).__init__()
        self.dataset_name = f"damped_harmonic_oscillator_{dim}"
        s0 = torch.normal(
            3.0, 0.2, size=(self.BATCH_SIZE, dim), device=train_device.get_device()
        )
        v0 = torch.normal(
            -0.0, 0.2, size=(self.BATCH_SIZE, dim), device=train_device.get_device()
        )
        self.x0 = torch.cat((s0, v0), dim=-1)
        self.ode_class = DampedODE(
            HarmonicOscillatorODE(
                self.x0, batch_size=self.BATCH_SIZE, spring_constant=self.SPRING_CONSTANT
            )
        )


class HarmonicOscillatorODE(ODE):
    def __init__(self, x0: torch.Tensor, batch_size: int, spring_constant: float, dim: int = 1):
        super(HarmonicOscillatorODE, self).__init__()
        self.dim = dim

        spring_constants = -spring_constant * torch.ones(batch_size, self.dim)
        spring_constants = torch.diag_embed(
            spring_constants, offset=0, dim1=-2, dim2=-1
        )
        self.x0 = x0
        self.a = spring_constants.to(train_device.get_device())

        self.opts = OdeintOptions()
        self.opts.solver = "rk4"
        self.opts.step_size = 0.005

    def forward(self, t, x):
        q, p = torch.chunk(x, 2, -1)
        s = p
        v = (self.a @ q.unsqueeze(-1)).squeeze(-1)
        return torch.cat((s, v), dim=-1)

    def true_solution(self, t, x0):
        s0, v0 = torch.chunk(x0, 2, -1)
        omega = torch.sqrt(-torch.diagonal(self.a, dim1=1, dim2=2))
        phi = torch.arctan(-v0 / omega / s0)
        a = s0 / torch.cos(phi)
        s = a.unsqueeze(1) * torch.cos(
            omega.unsqueeze(1) * t.unsqueeze(0).unsqueeze(2) + phi.unsqueeze(1)
        )
        v = (
            -a.unsqueeze(1)
            * omega.unsqueeze(1)
            * torch.sin(
                omega.unsqueeze(1) * t.unsqueeze(0).unsqueeze(2) + phi.unsqueeze(1)
            )
        )
        return torch.cat((s, v), dim=-1)

    def hamiltonian(self, x: torch.Tensor):
        dim = x.size()[0]
        a = self.a[0:dim]
        a = -torch.diagonal(a, dim1=1, dim2=2)
        s, v = torch.chunk(x, 2, -1)
        h = (
            torch.sum(a.unsqueeze(1) * torch.square(s), -1) / 2.0
            + torch.sum(torch.square(v), -1) / 2.0
        )
        return h


if __name__ == "__main__":
    harmonic_oscillator = HarmonicOscillatorFull()
    harmonic_oscillator.generate_and_save_data()
    harmonic_oscillator = HarmonicOscillatorThreeQuarter()
    harmonic_oscillator.generate_and_save_data()
