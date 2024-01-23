# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Bayesian Neural Network
================================
We demonstrate how to use NUTS to do inference on a simple (small)
Bayesian neural network with two hidden layers.
.. image:: ../_static/img/examples/bnn.png
    :align: center
"""

import os
import time
from typing import Tuple

import jax.random as random
import numpy as np
import jax.numpy as jnp
import numpyro
import torch
from numpyro.infer import MCMC, NUTS

from data.find_dataset_class import find_dataset_class
from mcmc_time_series.aphinity_warpper import get_aphinity_dataset
from mcmc_time_series.evaluate_mcmc import evaluate
from mcmc_time_series.models.model import bnn_odenet, aug_bnn_odenet
from mcmc_time_series.options.mcmc_options import MCMCOptions
from options.options_enum import DatasetEnum


def run_hmc_inference(
    model,
    opts: MCMCOptions,
    rng_key,
    t: jnp.ndarray,
    x0: jnp.ndarray,
    x: jnp.ndarray,
    hidden_dim: int,
):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=opts.num_warmup,
        num_samples=opts.num_samples,
        num_chains=opts.num_chains,
        progress_bar=True,
    )
    mcmc.run(rng_key, t, x0, x, hidden_dim)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


def get_data(dataset: DatasetEnum) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset_class = find_dataset_class(dataset)
    data = dataset_class().get_dataset("train")
    t = data.t.detach().cpu().numpy()
    x0 = data.x0.detach().cpu().numpy()
    x = data.x.detach().cpu().numpy()
    data = dataset_class().get_dataset("test")
    t_test = data.t.numpy()
    return t, x0, x, t_test


def main(opts: MCMCOptions):
    numpyro.set_platform(opts.use_gpu)
    numpyro.set_host_device_count(opts.num_chains)
    x0, x, t, t_test = get_aphinity_dataset(opts.dataset)
    x = x[0:opts.batch_size, :]
    x0 = x0[:opts.batch_size, :]
    x = jnp.array(x)
    x0 = jnp.array(x0)
    t = jnp.array(t)
    t_test = jnp.array(t_test)
    model = bnn_odenet
    if opts.use_aug_dynamics:
        model = aug_bnn_odenet

    if opts.run_mcmc:
        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        samples = run_hmc_inference(
            model, opts, rng_key, t, x0, x, hidden_dim=opts.hidden_dim
        )

        torch.save(
            {"samples": samples, "rng_key": rng_key, "rng_key_predict": rng_key_predict},
            os.path.join(opts.checkpoints_dir, "samples.pt"),
        )

    x0, x, t, t_test = get_aphinity_dataset(opts.dataset)
    x = jnp.array(x)
    x0 = jnp.array(x0)
    t = jnp.array(t)
    t_test = jnp.array(t_test)
    data = torch.load(os.path.join(opts.checkpoints_dir, "samples.pt"))
    samples = data["samples"]
    rng_key = data["rng_key"]
    rng_key_predict = data["rng_key_predict"]

    evaluate(model, x, x0, opts, rng_key_predict, samples, t, t_test)


