from pathlib import Path

from torch import optim

from aphynity.evaluate import evaluate
from aphynity.experiment import APHYNITYExperiment
from util.device_setter import train_device

from aphynity.aphinity_options.aphinityoptions import (
    DatasetEnum,
    PhyEnum,
    AphinityOptions,
)
from aphynity.datasets import init_dataloaders
from aphynity.forecasters import Forecaster
from aphynity.networks import (
    ReactionDiffusionParamPDE,
    ConvNetEstimator,
    DampedWaveParamPDE,
    DampedPendulumParamPDE,
    MLP,
    ZeroPredictor,
)
from aphynity.utils import init_weights


def run(
    opts: AphinityOptions,
    do_train_model: bool = True,
    do_fit_laplace: bool = True,
    do_generate_data: bool = True,
):
    data_path = Path(__file__).parent.parent / "datasets"
    train, test, train_data = init_dataloaders(opts.dataset, data_path)

    net = find_network_architecture(opts, train)

    optimizer = optim.Adam(net.parameters(), lr=opts.tau_1, betas=(0.9, 0.999))
    experiment = APHYNITYExperiment(
        train=train,
        test=test,
        net=net,
        optimizer=optimizer,
        min_op=opts.min_op,
        lambda_0=opts.lambda_0,
        tau_2=opts.tau_2,
        niter=opts.niter,
        nlog=10,
        nupdate=100,
        nepoch=opts.nepoch,
        path=opts.experiment_dir,
        device=train_device.get_device(),
    )
    if do_train_model:
        experiment.run()
        experiment.save(opts, train_data)
    evaluate(
        opts,
        net=experiment.net,
        train_data=train_data,
        do_fit_laplace=do_fit_laplace,
        do_generate_data=do_generate_data,
    )


def find_network_architecture(opts, train):
    if opts.dataset == DatasetEnum.rd:
        net = find_rd_architecture(opts, train)

    elif opts.dataset == DatasetEnum.wave:
        net = find_wave_architecture(opts, train)

    elif opts.dataset == DatasetEnum.pendulum:
        net = find_pendulum_architecture(opts, train)
    else:
        raise KeyError(f"Dataset {opts.dataset} not found")
    return net


def find_pendulum_architecture(opts, train):
    if opts.phy == PhyEnum.none:
        model_phy = ZeroPredictor()
    elif opts.phy == PhyEnum.incomplete:
        model_phy = DampedPendulumParamPDE(is_complete=False, real_params=None)
    elif opts.phy == PhyEnum.complete:
        model_phy = DampedPendulumParamPDE(is_complete=True, real_params=None)
    elif opts.phy == PhyEnum.true:
        model_phy = DampedPendulumParamPDE(
            is_complete=True, real_params=train.dataset.params
        )
    else:
        raise KeyError(f"Phy model {opts.phy} not found")
    model_aug = MLP(state_c=opts.state_c, hidden=opts.hidden)
    init_weights(model_aug, init_type="orthogonal", init_gain=0.2)
    net = Forecaster(model_phy=model_phy, model_aug=model_aug, is_augmented=opts.aug)
    return net


def find_wave_architecture(opts, train):
    if opts.phy == PhyEnum.none:
        model_phy = ZeroPredictor()
    elif opts.phy == PhyEnum.incomplete:
        model_phy = DampedWaveParamPDE(is_complete=False, real_params=None)
    elif opts.phy == PhyEnum.complete:
        model_phy = DampedWaveParamPDE(is_complete=True, real_params=None)
    elif opts.phy == PhyEnum.true:
        model_phy = DampedWaveParamPDE(
            is_complete=True, real_params=train.dataset.params
        )
    else:
        raise KeyError(f"Phy model {opts.phy} not found")
    model_aug = ConvNetEstimator(state_c=opts.state_c, hidden=opts.hidden)
    net = Forecaster(model_phy=model_phy, model_aug=model_aug, is_augmented=opts.aug)
    return net


def find_rd_architecture(opts, train):
    if opts.phy == PhyEnum.incomplete:
        model_phy = ReactionDiffusionParamPDE(
            dx=train.dataset.dx, is_complete=False, real_params=None
        )
    elif opts.phy == PhyEnum.complete:
        model_phy = ReactionDiffusionParamPDE(
            dx=train.dataset.dx, is_complete=True, real_params=None
        )
    elif opts.phy == PhyEnum.true:
        model_phy = ReactionDiffusionParamPDE(
            dx=train.dataset.dx, is_complete=True, real_params=train.dataset.params
        )
    else:
        raise KeyError(f"Phy model {opts.phy} not found")
    model_aug = ConvNetEstimator(state_c=opts.state_c, hidden=opts.hidden)
    net = Forecaster(model_phy=model_phy, model_aug=model_aug, is_augmented=opts.aug)
    return net
