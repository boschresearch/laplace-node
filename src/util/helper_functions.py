import os
from typing import List, Union, Iterable, Generator


def mkdirs(paths: Union[List[str], str]):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def inf_generator(iterable: Iterable) -> Generator:
    """Allows training with DataLoaders in a single infinite loop:
       Usage: for i, (x, y) in enumerate(inf_generator(train_loader))
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()



