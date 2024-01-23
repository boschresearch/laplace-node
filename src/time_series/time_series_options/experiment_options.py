import os
from pathlib import Path

from options.options_base import OptionsBase
from options.options_enum import OptimizerEnum, DatasetEnum, NetworkEnum, ActivationEnum
from time_series.models.odeint_options import OdeintOptions

dirname = os.path.dirname(__file__)


class ExperimentOptions(OptionsBase):
    odeint_options: OdeintOptions = OdeintOptions()
    lr: float = 1e-1
    batch_size: int = 1
    optimizer: OptimizerEnum = OptimizerEnum.adam
    name: str = "neural_ode"
    dataset: DatasetEnum = DatasetEnum.trigonometric_data
    output_dir: str = str(Path(__file__).parent.parent.parent / "experiments")
    checkpoint_freq: int = 500
    niter: int = 1000
    use_gpu: bool = False
    network: NetworkEnum = NetworkEnum.naive
    use_dissipation: bool = False
    hidden_dim: int = 6
    act: ActivationEnum = ActivationEnum.tanh
    random_seed: int = 0

    run_experiment: bool = True
    run_laplace: bool = True
    evaluate_experiment: bool = True
