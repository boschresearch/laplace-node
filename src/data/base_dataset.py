import os
from typing import Union

from torch.utils import data as data
from torch.utils.data import Dataset

from data.time_series_data.toy_timeseriers_dataset import TimeSeriesDataset


class BaseDataset:
    DIMENSION = None
    NUM_IP_CH = None
    NUM_CLASSES = None

    def __init__(self):
        self.path_to_data = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "datasets"
        )

    def return_dataloader(
        self,
        phase: str,
        batch_size: int,
        shuffle: bool,
        num_threads: int,
    ) -> data.DataLoader:
        dataset = self.get_dataset(phase)
        dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_threads,
            drop_last=phase == "train",
        )
        return dataloader

    def get_dataset(self, split: str) -> Union[Dataset, TimeSeriesDataset]:
        raise not NotImplementedError
