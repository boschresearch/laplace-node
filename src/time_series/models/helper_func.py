from typing import Union

import torch
import torch.nn as nn

from options.options_enum import ActivationEnum


def get_activation(act=ActivationEnum.relu) -> Union[nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.Hardtanh, nn.ReLU6]:
    if act == ActivationEnum.relu:
        return nn.ReLU(inplace=True)
    if act == ActivationEnum.lrelu:
        return nn.LeakyReLU()
    if act == ActivationEnum.sigmoid:
        return nn.Sigmoid()
    if act == ActivationEnum.tanh:
        return nn.Tanh()
    if act == ActivationEnum.hardtanh:
        return nn.Hardtanh()
    if act == ActivationEnum.relu6:
        return nn.ReLU6()
    if act == ActivationEnum.log_cosh:
        return log_cosh
    raise NotImplementedError("Activation function {} not implemented".format(act))


def log_cosh(x):
    return torch.log(torch.cosh(x))

