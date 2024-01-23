from copy import deepcopy

import torch
from laplace import FullLaplace
from laplace.curvature import GGNInterface

from extend_laplace_torch.jacobian import manual_jacobian

"""
The following function is adapted from Laplace Version 0.1a1
( https://github.com/AlexImmer/Laplace/releases/tag/0.1a1
Copyright (c) 2021 Alex Immer, licensed under the MIT License,
cf. 3rd-party-license.txt file in the root directory of this source tree).
"""


class ManualGGN(GGNInterface):

    def kron(self, x, y, **kwargs):
        raise NotImplementedError("Kronecker-factorized Hessian not yet implemented for ManualGGN")

    def diag(self, x, y, **kwargs):
        raise NotImplementedError("Diagonal Hessian not yet implemented for ManualGGN")

    def jacobians(self, x: torch.Tensor):
        return manual_jacobian(self.model, x)

    def gradients(self, x, y):
        f = self.model(x)
        loss = self.lossfunc(f, y)
        loss.backward()
        Gs = torch.cat([p.grad_batch.data.flatten(start_dim=1)
                        for p in self._model.parameters()], dim=1)
        return Gs, loss


def _precision_to_scale_tril(P):
    # Ref: https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
    flipped_P = torch.flip(P, (-2, -1))

    Lf = stable_cholesky(flipped_P)

    L_inv = torch.transpose(torch.flip(Lf, (-2, -1)), -2, -1)
    L = torch.triangular_solve(torch.eye(P.shape[-1], dtype=P.dtype, device=P.device),
                               L_inv, upper=False)[0]
    return L


def stable_cholesky(M: torch.Tensor):
    M_old = deepcopy(M.detach())
    jitter = 0.0001
    max_jitter = 1e10
    has_cholesky_factorization = False
    while not has_cholesky_factorization and jitter < max_jitter:
        try:
            Lf = torch.linalg.cholesky(M)
            has_cholesky_factorization = True
        except (ValueError, RuntimeError) as e:
            print(f"Adding jitter to M: {jitter}")
            M, jitter = add_jitter(M, M_old, jitter)

    if not has_cholesky_factorization:
        raise ValueError("Variable flipped_P does not have a Cholesky factorization. "
                         "Consider raising the max_jitter value")
    return Lf


def add_jitter(M: torch.Tensor, M_old: torch.Tensor, jitter: float):
    M = M_old + torch.diag(torch.ones(M.shape[0], device=M.device)) * jitter
    jitter = jitter * 10.
    return M, jitter


def invsqrt_precision(M):
    """Compute ``M^{-0.5}`` as a tridiagonal matrix.

    Parameters
    ----------
    M : torch.Tensor

    Returns
    -------
    M_invsqrt : torch.Tensor
    """
    return _precision_to_scale_tril(M)


class FullLaplace2(FullLaplace):

    def _compute_scale(self):
        self._posterior_scale = invsqrt_precision(self.posterior_precision)

    def functional_variance(self, Js):
        func_var = torch.einsum('ncp,pq,nkq->nck', Js, self.posterior_covariance, Js)
        return func_var
