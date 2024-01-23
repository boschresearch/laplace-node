from jax._src.api import vmap, jit
import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions
from numpyro import handlers
from jax.experimental.ode import odeint


def neural_net(x, time, params):
    """
    Current implementation does not use time as an input but odeint expects time input
    """
    w1, w2, w3 = params
    z1 = act(jnp.matmul(x, w1) + w2)
    z3 = jnp.matmul(z1, w3)
    return z3


def pendulum_dynamics(x, time, params):
    omega = params
    s = x[0]
    v = x[1][None]
    dsdt = v
    dpdt = -omega * jnp.sin(s)
    return jnp.concatenate((dsdt, dpdt))


def aug_model(x, time, params):
    w1, w2, w3, omega = params
    aug_out = pendulum_dynamics(x, time, omega)
    nn_out = neural_net(x, time, (w1, w2, w3))
    return aug_out + nn_out


def odenet(params, X0, t):
    final_state = odeint(neural_net, X0, t, params)
    return final_state

def aug_odenet(params, X0, t):
    final_state = odeint(aug_model, X0, t, params)
    return final_state

@jax.jit
def batched_odenet(params, X0, t):
    mapped_model = vmap(odenet, in_axes=(None, 0, None))
    result = mapped_model(params, X0, t)
    return result

@jax.jit
def batched_aug_odenet(params, X0, t):
    mapped_model = vmap(aug_odenet, in_axes=(None, 0, None))
    result = mapped_model(params, X0, t)
    return result


def bnn_odenet(t, X0, X, hidden_dim: int):
    w1, w2, w3 = sample_model(in_dim=X0.shape[-1], hidden_dim=hidden_dim)
    final_state = batched_odenet((w1, w2, w3), X0, t)
    # we put a prior on the observation noise
    prec_obs = numpyro.sample("prec_obs", distributions.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)
    # observe data
    Y = numpyro.sample("Y", distributions.Normal(final_state, sigma_obs), obs=X)
    return Y


def aug_bnn_odenet(t, X0, X, hidden_dim: int):
    w1, w2, w3, omega = sample_aug_model(in_dim=X0.shape[-1], hidden_dim=hidden_dim)
    final_state = batched_aug_odenet((w1, w2, w3, omega), X0, t)
    # we put a prior on the observation noise
    prec_obs = numpyro.sample("prec_obs", distributions.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)
    # observe data
    Y = numpyro.sample("Y", distributions.Normal(final_state, sigma_obs), obs=X)
    return Y


def act(x):
    return jnp.tanh(x)


def sample_aug_model(in_dim: int, hidden_dim: int):
    w1, w2, w3 = sample_model(in_dim, hidden_dim)
    omega = numpyro.sample(
        "omega", distributions.Normal(jnp.zeros((1,)), jnp.ones((1,)))
    )
    return w1, w2, w3, omega


def sample_model(in_dim: int, hidden_dim: int):
    w1 = numpyro.sample(
        "w1",
        distributions.Normal(
            jnp.zeros((in_dim, hidden_dim)), jnp.ones((in_dim, hidden_dim))
        ),
    )
    w2 = numpyro.sample(
        "w2", distributions.Normal(jnp.zeros((hidden_dim,)), jnp.ones((hidden_dim,)))
    )
    w3 = numpyro.sample(
        "w3",
        distributions.Normal(
            jnp.zeros((hidden_dim, in_dim)), jnp.ones((hidden_dim, in_dim))
        ),
    )
    return w1, w2, w3


def predict(model, rng_key, samples, t, x0, hidden_dim: int):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    model_trace = handlers.trace(model).get_trace(
        t=t, X0=x0, X=None, hidden_dim=hidden_dim
    )
    return model_trace["Y"]["value"]


def sample_and_model(x, in_dim: int, hidden_dim: int):
    w1, w2, w3 = sample_model(in_dim, hidden_dim)
    return neural_net(x, None, (w1, w2, w3))


def predict_odefunc(rng_key, samples, x, in_dim: int, hidden_dim: int):
    model = handlers.substitute(handlers.seed(sample_and_model, rng_key), samples)
    return model(x, in_dim, hidden_dim)

def sample_and_model_aug(x, in_dim: int, hidden_dim: int):
    w1, w2, w3, omega = sample_aug_model(in_dim, hidden_dim)
    return aug_model(x, None, (w1, w2, w3, omega))


def predict_odefunc_aug(rng_key, samples, x, in_dim: int, hidden_dim: int):
    model = handlers.substitute(handlers.seed(sample_and_model_aug, rng_key), samples)
    return model(x, in_dim, hidden_dim)
