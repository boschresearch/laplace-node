import torch


class Device:
    def __init__(self):
        self._device = "cpu"

    def set_device(self, device: torch.device):
        self._device = device

    def get_device(self):
        return self._device


train_device = Device()
