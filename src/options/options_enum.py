from enum import Enum


class OptimizerEnum(str, Enum):
    sgd = "sgd"
    adam = "adam"


class DatasetEnum(str, Enum):
    harmonic_oscillator = "harmonic_oscillator"
    harmonic_oscillator_half = "harmonic_oscillator_half"
    harmonic_oscillator_three_quarter = "harmonic_oscillator_three_quarter"
    damped_harmonic_oscillator = "damped_harmonic_oscillator"
    pendulum = "pendulum"
    lotka_volterra_half = "lotka_volterra_half"
    lotka_volterra_full = "lotka_volterra_full"
    lotka_volterra_poly = "lotka_volterra_poly"
    trigonometric_data = "trigonometric_data"


class NetworkEnum(str, Enum):
    naive = "naive"
    hamiltonian = "hamiltonian"
    separable_hamiltonian = "separable_hamiltonian"
    newtonian_hamiltonian = "newtonian_hamiltonian"
    constrained_hamiltonian = "constrained_hamiltonian"


class ActivationEnum(str, Enum):
    tanh = "tanh"
    relu = "relu"
    lrelu = "lrelu"
    sigmoid = "sigmoid"
    hardtanh = "hardtanh"
    relu6 = "relu6"
    log_cosh = "log_cosh"


class AutodifEnum(str, Enum):
    adjoint = "adjoint"
    naive = "naive"


class SolverEnum(str, Enum):
    euler = "euler"
    rk4 = "rk4"
    midpoint = "midpoint"
    symplectic_euler = "symplectic_euler"
    dopri5 = "dopri5"
