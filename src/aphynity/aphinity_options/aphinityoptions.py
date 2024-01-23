from enum import Enum
from pathlib import Path

from options.options_base import OptionsBase


class DatasetEnum(str, Enum):
    rd = "rd"
    wave = "wave"
    pendulum = "pendulum"


class PhyEnum(str, Enum):
    incomplete = "incomplete"
    complete = "complete"
    true = "true"
    none = "none"


class RegularizerEnum(str, Enum):
    l2_normalized = "l2_normalized"
    l2 = "l2"
    none = "none"


class AphinityOptions(OptionsBase):
    dataset: DatasetEnum = DatasetEnum.pendulum
    phy: PhyEnum = PhyEnum.incomplete
    output_dir: str = str(Path(__file__).parent.parent.parent / "experiments")
    aug: bool = True
    name: str = "aphynity"
    use_gpu: bool = False
    lambda_0: float = 1.0
    tau_1: float = 1e-3
    tau_2: float = 1
    niter: int = 5
    min_op: str = RegularizerEnum.none
    nepoch: int = 1
    hidden: int = 8
    state_c: int = 2
