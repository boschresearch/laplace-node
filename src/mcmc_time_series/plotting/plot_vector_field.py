import os
import sys

if sys.platform == "linux":
    from jax._src.api import vmap
    from mcmc_time_series.models.model import predict_odefunc, predict_odefunc_aug
import numpy as np
import torch

from time_series.plotting.plot_vector_field_uncertainty import (
    VectorFieldPlotUncertainty,
)


class MCMCVectorFieldPlotUncertainty(VectorFieldPlotUncertainty):
    NAME = "mcmc_vector_field_uncertainty"

    def data(self, x_train: np.ndarray, N: int, hidden_dim: int, vmap_args, batch_size: int, use_aug_model):
        X = np.linspace(-7, 12, N)
        Y = np.linspace(-2, 2.5, N)
        U, V = np.meshgrid(X, Y)
        shape_u = U.shape
        U = U.flatten()[:, np.newaxis]
        V = V.flatten()[:, np.newaxis]
        data = np.concatenate((U, V), axis=1)
        print(data.shape)
        if use_aug_model:
            predict_function = predict_odefunc_aug
        else:
            predict_function = predict_odefunc

        vec = None
        colorcode = []
        for i in range(data.shape[0]):
            if vec is None:
                x = data[i]
                predictions = vmap(
                    lambda samples, rng_key: predict_function(
                        rng_key, samples, x, in_dim=x.shape[0], hidden_dim=hidden_dim
                    )
                )(*vmap_args)
                vec = np.mean(predictions, axis=0)[np.newaxis]
                cov = np.mean(np.std(predictions, axis=0))
                colorcode.append(cov)

            else:
                x = data[i]
                predictions = vmap(
                    lambda samples, rng_key: predict_function(
                        rng_key, samples, x, in_dim=x.shape[0], hidden_dim=hidden_dim
                    )
                )(*vmap_args)
                v = np.mean(predictions, axis=0)
                cov = np.mean(np.std(predictions, axis=0))
                colorcode.append(cov)

                vec = np.concatenate((vec, v[np.newaxis]), axis=0,)
        U = vec[:, 0].reshape(shape_u)
        V = vec[:, 1].reshape(shape_u)
        colorcode = np.array(colorcode).reshape(shape_u)
        x_train = x_train[0:batch_size]
        torch.save(
            {"U": U, "V": V, "X": X, "Y": Y, "var": colorcode, "x_train": x_train},
            os.path.join(self.fig_data, f"{self.NAME}.pt"),
        )
