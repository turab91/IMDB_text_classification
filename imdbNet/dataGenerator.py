import numpy as np
from numpy import ndarray
from typing import Iterator


class DataGenerator:
    def __init__(self, inputs: ndarray, targets: ndarray, batch_size: int, shuffle: bool = False) -> Iterator:
        r"""
        construct a  object to iterate over inputs and targets
        """
        assert inputs.shape[0] == targets.shape[0], \
            f"inputs and targets should be of same size. Got {inputs.shape[0]} != {targets.shape[0]}"

        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._num_examples = inputs.shape[0]
        self._num_batch = np.ceil(self._num_examples / self.batch_size).astype(np.int32)

    def __iter__(self):
        r"""
        iterate over inputs and targets in batches
        """
        idx = np.arange(self._num_examples)
        if self.shuffle:
            np.random.shuffle(idx)
        return iter((self.inputs[mask], self.targets[mask]) for mask in np.array_split(idx, self._num_batch))

    @property
    def num_batch(self):
        return self._num_batch

