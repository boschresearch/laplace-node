import os

from torch.utils.data import DataLoader

from .pendulum import DampledPendulum
from .rd import ReactionDiffusion
from .wave import DampedWaveEquation


def param_rd(buffer_filepath, batch_size=64):
    dataset_train_params = {
        "path": os.path.join(buffer_filepath, "train"),
        "group": "train",
        "num_seq": 1600,
        "size": 32,
        "time_horizon": 3,
        "dt": 0.1,
        "group": "train",
    }

    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["num_seq"] = 320
    dataset_test_params["group"] = "test"
    dataset_test_params["path"] = os.path.join(buffer_filepath, "test"),

    dataset_train = ReactionDiffusion(**dataset_train_params)
    dataset_test = ReactionDiffusion(**dataset_test_params)

    dataloader_train_params = {
        "dataset": dataset_train,
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": True,
        "drop_last": False,
        "shuffle": True,
    }

    dataloader_test_params = {
        "dataset": dataset_test,
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": True,
        "drop_last": False,
        "shuffle": False,
    }

    dataloader_train = DataLoader(**dataloader_train_params)
    dataloader_test = DataLoader(**dataloader_test_params)

    return dataloader_train, dataloader_test


def param_wave(buffer_filepath, batch_size=16):
    dataset_train_params = {
        "path": os.path.join(buffer_filepath, "train"),
        "size": 64,
        "seq_len": 25,
        "dt": 1e-3,
        "group": "train",
        "num_seq": 200,
    }

    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["num_seq"] = 50
    dataset_test_params["group"] = "test"
    dataset_test_params["path"] = os.path.join(buffer_filepath, "test")

    dataset_train = DampedWaveEquation(**dataset_train_params)
    dataset_test = DampedWaveEquation(**dataset_test_params)

    dataloader_train_params = {
        "dataset": dataset_train,
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": True,
        "drop_last": False,
        "shuffle": True,
    }

    dataloader_test_params = {
        "dataset": dataset_test,
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": True,
        "drop_last": False,
        "shuffle": False,
    }
    dataloader_train = DataLoader(**dataloader_train_params)
    dataloader_test = DataLoader(**dataloader_test_params)

    return dataloader_train, dataloader_test, dataset_train


def param_pendulum(buffer_filepath, batch_size=25):
    dataset_train_params = {
        "num_seq": 25,
        "time_horizon": 10,
        "dt": 0.5,
        "group": "train",
        "path": os.path.join(buffer_filepath, "train"),
    }

    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["num_seq"] = 25
    dataset_test_params["group"] = "test"
    dataset_test_params["path"] = os.path.join(buffer_filepath, "test")

    dataset_train = DampledPendulum(**dataset_train_params)
    dataset_test = DampledPendulum(**dataset_test_params)

    dataloader_train_params = {
        "dataset": dataset_train,
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": True,
        "drop_last": False,
        "shuffle": True,
    }

    dataloader_test_params = {
        "dataset": dataset_test,
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": True,
        "drop_last": False,
        "shuffle": False,
    }
    dataloader_train = DataLoader(**dataloader_train_params)
    dataloader_test = DataLoader(**dataloader_test_params)

    return dataloader_train, dataloader_test, dataset_train


def init_dataloaders(dataset, buffer_filepath=None):
    assert buffer_filepath is not None
    if dataset == "rd":
        buffer_filepath = os.path.join(buffer_filepath, "aphinity_rd")
        os.makedirs(buffer_filepath, exist_ok=True)
        return param_rd(buffer_filepath)
    elif dataset == "wave":
        buffer_filepath = os.path.join(buffer_filepath, "aphinity_wave")
        os.makedirs(buffer_filepath, exist_ok=True)
        return param_wave(buffer_filepath)
    elif dataset == "pendulum":
        buffer_filepath = os.path.join(buffer_filepath, "aphinity_pendulum")
        os.makedirs(buffer_filepath, exist_ok=True)
        return param_pendulum(buffer_filepath)
