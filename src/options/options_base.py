import os

import torch
import yaml
from pydantic.main import BaseModel
from util.device_setter import train_device


class OptionsBase(BaseModel):
    use_gpu: bool
    output_dir: str
    name: str
    experiment_dir: str = None
    checkpoints_dir: str = None

    class Config:
        use_enum_values = True

    def __init__(self, **kwargs):
        super(OptionsBase, self).__init__(**kwargs)
        # self.initialize_setup()

    def initialize_setup(self):
        self.experiment_dir = os.path.join(self.output_dir, self.name)
        self.checkpoints_dir = os.path.join(self.experiment_dir, "checkpoints")
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self._set_gpu_options()
        self.save_options()

    def _set_gpu_options(self):
        if self.use_gpu and torch.cuda.is_available():
            train_device.set_device(torch.device("cuda"))
        else:
            train_device.set_device("cpu")

    def save_options(self):
        file_path = os.path.join(self.experiment_dir, "aphinity_options")
        os.makedirs(file_path, exist_ok=True)
        with open(os.path.join(file_path, "opts.yaml"), "w") as outfile:
            yaml.dump(self.dict(), outfile, default_flow_style=False)
