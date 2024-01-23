from pathlib import Path

from options.options_base import OptionsBase
from options.options_enum import DatasetEnum


class MCMCOptions(OptionsBase):
    name: str = "neural_ode"
    dataset: DatasetEnum = DatasetEnum.pendulum
    output_dir: str = str(Path(__file__).parent.parent.parent / "experiments")
    hidden_dim: int = 10
    num_chains: int = 1
    num_warmup: int = 50
    num_samples: int = 500
    batch_size: int = 2
    use_gpu = False
    experiment_dir: str = None
    checkpoints_dir: str = None

    use_aug_dynamics: bool = True

    run_mcmc: bool = True

    class Config:
        use_enum_values = True

