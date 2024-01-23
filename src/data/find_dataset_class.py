from typing import Type, Union

from data.base_dataset import BaseDataset
from data.time_series_data.harmonic_oscillator_dataset import HarmonicOscillator, HarmonicOscillatorHalf, \
    HarmonicOscillatorThreeQuarter, DampedHarmonicOscillator
from data.time_series_data.lotka_volterra_dataset import LotkaVolterraFull, LotkaVolterraHalf
from data.time_series_data.lotka_volterra_poly_dataset import LotkaVolterraPoly
from data.time_series_data.pendulum_dataset import Pendulum
from data.time_series_data.trigonomic_dataset import TrigonometricData
from options.options_enum import DatasetEnum


def find_dataset_class(
    dataset_name: str,
) -> Type[Union[BaseDataset]]:
    dataset_name = dataset_name.lower()
    if dataset_name == DatasetEnum.harmonic_oscillator:
        return HarmonicOscillator
    if dataset_name == DatasetEnum.harmonic_oscillator_half:
        return HarmonicOscillatorHalf
    if dataset_name == DatasetEnum.harmonic_oscillator_three_quarter:
        return HarmonicOscillatorThreeQuarter
    if dataset_name == DatasetEnum.damped_harmonic_oscillator:
        return DampedHarmonicOscillator
    if dataset_name == DatasetEnum.pendulum:
        return Pendulum
    if dataset_name == DatasetEnum.lotka_volterra_full:
        return LotkaVolterraFull
    if dataset_name == DatasetEnum.lotka_volterra_half:
        return LotkaVolterraHalf
    if dataset_name == DatasetEnum.lotka_volterra_poly:
        return LotkaVolterraPoly
    if dataset_name == DatasetEnum.trigonometric_data:
        return TrigonometricData
    else:
        raise NotImplementedError(f"The dataset {dataset_name} is not implemented!")
