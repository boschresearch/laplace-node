import torch.nn as nn
from torchdiffeq import odeint

"""
The following code is adapted from APHYNITY Commit 1b0fcfc7
( https://github.com/yuan-yin/APHYNITY
Copyright (c) 2021 Yuan Yin, Vincent Le Guen, Jérémie Dona, licensed under the MIT License,
cf. 3rd-party-license.txt file in the root directory of this source tree)
"""


class DerivativeEstimator(nn.Module):
    def __init__(self, model_phy, model_aug, is_augmented):
        super().__init__()
        self.model_phy = model_phy
        self.model_aug = model_aug
        self.is_augmented = is_augmented

    def forward(self, t, state):
        res_phy = self.model_phy(state)
        if self.is_augmented:
            res_aug = self.model_aug(state)
            return res_phy + res_aug
        else:
            return res_phy


class Forecaster(nn.Module):
    def __init__(self, model_phy, model_aug, is_augmented, method="dopri5"):
        super().__init__()

        self.model_phy = model_phy
        self.model_aug = model_aug

        self.derivative_estimator = DerivativeEstimator(
            self.model_phy, self.model_aug, is_augmented=is_augmented
        )
        self.method = method
        self.options = None
        self.int_ = odeint

    def forward(self, y0, t):
        # y0 = y[:,:,0]
        res = odeint(
            self.derivative_estimator,
            y0=y0,
            t=t,
            atol=1e-7,
            rtol=1e-5,
            method=self.method,
            options=self.options,
        )
        # res: T x batch_size x n_c (x h x w)
        dim_seq = y0.dim() + 1
        dims = [1, 2, 0] + list(range(dim_seq))[3:]
        return res.permute(*dims)  # batch_size x n_c x T (x h x w)

    def get_pde_params(self):
        return self.model_phy.params
