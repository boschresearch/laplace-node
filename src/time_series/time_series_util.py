from options.options_enum import NetworkEnum
from time_series.models.model import (
    NeuralNetwork,
    Hamiltonian,
    HamiltonianNeuralNetwork,
    SeparableHamiltonian,
    ConstrainedHamiltonian,
    PSDNetwork,
    NewtonianHamiltonian,
)
from time_series.time_series_options.experiment_options import ExperimentOptions


def find_network(dataset, opts: ExperimentOptions):
    if opts.network == NetworkEnum.naive:
        return NeuralNetwork(
            dataset.DIMENSION,
            out_dim=dataset.DIMENSION,
            hidden_dim=opts.hidden_dim,
            act=opts.act,
        )
    if opts.network in [network.value for network in NetworkEnum]:
        if opts.network == NetworkEnum.hamiltonian:
            h = Hamiltonian(dataset.DIMENSION, opts.hidden_dim, act=opts.act)
        elif opts.network == NetworkEnum.separable_hamiltonian:
            h = SeparableHamiltonian(dataset.DIMENSION, opts.hidden_dim, act=opts.act)
        elif opts.network == NetworkEnum.constrained_hamiltonian:
            h = ConstrainedHamiltonian(dataset.DIMENSION, opts.hidden_dim, act=opts.act)
        elif opts.network == NetworkEnum.newtonian_hamiltonian:
            h = NewtonianHamiltonian(dataset.DIMENSION, opts.hidden_dim, act=opts.act)
        if opts.use_dissipation:
            dissipation_network = PSDNetwork(
                int(dataset.DIMENSION / 2),
                opts.hidden_dim,
                act=opts.act,
            )
            return HamiltonianNeuralNetwork(
                in_dim=dataset.DIMENSION,
                hamiltonian=h,
                dissipation_network=dissipation_network,
            )
        return HamiltonianNeuralNetwork(in_dim=dataset.DIMENSION, hamiltonian=h)
    else:
        raise NotImplementedError(f"The option: {opts.network} is not implemented. It has to be one of"
                                  f"{[network.value for network in NetworkEnum]}")
