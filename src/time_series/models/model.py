from typing import Optional, Union

import torch
from torch.nn import init

from time_series.models.helper_func import get_activation
from util.device_setter import train_device


class NeuralNetwork(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, last_layer_bias: bool = True, act: str = "tanh"):
        super(NeuralNetwork, self).__init__()
        self.lin1 = torch.nn.Linear(in_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, out_dim, bias=last_layer_bias)
        for l in [self.lin1, self.lin2]:
            init.orthogonal_(l.weight)
        self.activation_function = get_activation(act)

    def forward(self, t, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.activation_function(x)
        x = self.lin2(x)
        return x


class SeparableHamiltonian(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, act: str = "tanh"):
        super(SeparableHamiltonian, self).__init__()
        in_dim = int(in_dim / 2)
        self.dim = in_dim
        self.T = NeuralNetwork(in_dim, out_dim=1, hidden_dim=hidden_dim, last_layer_bias=False, act=act)
        self.V = NeuralNetwork(in_dim, out_dim=1, hidden_dim=hidden_dim, last_layer_bias=False, act=act)

    def forward(self, t, x: torch.Tensor) -> torch.Tensor:
        q = x[:, : self.dim]
        p = x[:, self.dim :]
        return self.V(t, q) + self.T(t, p)


class ConstrainedHamiltonian(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, act: str = "tanh"):
        super(ConstrainedHamiltonian, self).__init__()
        in_dim = int(in_dim / 2)
        self.dim = in_dim

        self.V = NeuralNetwork(in_dim, out_dim=1, hidden_dim=hidden_dim, act=act, last_layer_bias=False)
        self.M_inv = PSDNetwork(in_dim, hidden_dim, act = act)

    def T(self, t, q: torch.Tensor, p: torch.Tensor):
        # Consider generally changing the shape of the data to B x 1 x D to avoid squeezing
        return (p.unsqueeze(1) @ self.M_inv(t, q) @ p.unsqueeze(-1)).squeeze(-1)

    def forward(self, t, x: torch.Tensor) -> torch.Tensor:
        q = x[:, :self.dim]
        p = x[:, self.dim :]
        return self.V(t, q) + self.T(t, q, p)


class NewtonianHamiltonian(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, act: str = "tanh"):
        super(NewtonianHamiltonian, self).__init__()
        in_dim = int(in_dim / 2)
        self.dim = in_dim

        self.V = NeuralNetwork(in_dim, out_dim=1, hidden_dim=hidden_dim, act=act, last_layer_bias=False)
        self.M_sqrt = torch.nn.Parameter(torch.empty((in_dim, in_dim), device=train_device.get_device()), requires_grad=True)
        init.orthogonal(self.M_sqrt)

    def T(self, t, p: torch.Tensor):
        M = (self.M_sqrt @ self.M_sqrt.T).unsqueeze(0)
        return 0.5*(p.unsqueeze(1) @ M  @ p.unsqueeze(-1)).squeeze(-1)

    def forward(self, t, x: torch.Tensor) -> torch.Tensor:
        q = x[:, :self.dim]
        p = x[:, self.dim :]
        return self.V(t, q) + self.T(t, p)


class Hamiltonian(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, act: str = "tanh"):
        super(Hamiltonian, self).__init__()
        self.H = NeuralNetwork(in_dim=in_dim, out_dim=1, hidden_dim=hidden_dim, last_layer_bias=False, act=act)

    def forward(self, t, x: torch.Tensor) -> torch.Tensor:
        return self.H(t, x)


class PSDNetwork(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: Union[int, None] = None, act: str = "tanh"):
        super(PSDNetwork, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        tri_dim = int(
            out_dim * (out_dim + 1) / 2
        )  # number of elements in a lower dimensional matrix of size in_dim x in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.upper_triangle = NeuralNetwork(in_dim=in_dim, out_dim=tri_dim, hidden_dim=hidden_dim, act=act)
        self.tril_indices = torch.tril_indices(row=out_dim, col=out_dim, offset=0)

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        x = self.upper_triangle(t, x)
        batch_dim = x.shape[0]
        self.m = torch.zeros((batch_dim, self.out_dim, self.out_dim), device=train_device.get_device())
        self.m[:, self.tril_indices[0], self.tril_indices[1]] = x
        jitter = 1e-5
        diag_offset = torch.diag_embed(torch.ones(batch_dim, self.in_dim, device=train_device.get_device())) * jitter
        return torch.bmm(self.m, self.m.transpose(1, 2)) + diag_offset


"""
The following code is adapted from hamiltonian-nn Commit bcc362235
( https://github.com/greydanus/hamiltonian-nn
licensed under the Apache 2.0 license,
cf. 3rd-party-license.txt file in the root directory of this source tree)
"""


class HamiltonianNeuralNetwork(torch.nn.Module):
    """Code adapted from HNN paper"""

    def __init__(
        self,
        in_dim,
        hamiltonian: torch.nn.Module,
        dissipation_network: Optional[PSDNetwork] = None,
        assume_canonical_coords=True,
    ):
        super(HamiltonianNeuralNetwork, self).__init__()
        self.hamiltonian = hamiltonian
        self.dissipation_network = dissipation_network
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(in_dim)  # Levi-Civita permutation tensor

    def forward(self, t, x: torch.Tensor):
        """
        :param t:
        :param x: vector of generalized coordinated where x = (q, p)
        :return: returns the time derivative of x i.e. dx_dt = (dq_dt, dp_dt)
        """
        with torch.enable_grad():
            x = x.requires_grad_()
            hamiltonian = self.hamiltonian(None, x)
            dx_dt = self.time_derivative(hamiltonian, x)
            if self.dissipation_network is not None:
                q, p = torch.chunk(x, 2, -1)
                dissipation_matrix = self.dissipation_network(t, q)
                dq_dt, dp_dt = torch.chunk(dx_dt, 2, -1)
                dp_dt = dp_dt - (dissipation_matrix @ dq_dt.unsqueeze(1)).reshape(p.shape)
                dx_dt = torch.cat((dq_dt, dp_dt), dim=-1)
            return dx_dt

    def time_derivative(self, hamiltonian: torch.tensor, x: torch.tensor):

        """NEURAL HAMILTONIAN-STLE VECTOR FIELD"""

        dh = torch.autograd.grad(hamiltonian.sum(), x, create_graph=True)[
            0
        ]  # gradients for conservative field
        conservative_field = dh @ self.M.t()

        return conservative_field

    def permutation_tensor(self, n):
        if self.assume_canonical_coords:
            M = torch.eye(n, device=train_device.get_device())
            M = torch.cat([M[n // 2 :], -M[: n // 2]])
        else:
            """Constructs the Levi-Civita permutation tensor"""
            M = torch.ones(n, n, device=train_device.get_device())  # matrix of ones
            M *= 1 - torch.eye(n)  # clear diagonals
            M[::2] *= -1  # pattern of signs
            M[:, ::2] *= -1

            for i in range(n):  # make asymmetric
                for j in range(i + 1, n):
                    M[i, j] *= -1
        return M
