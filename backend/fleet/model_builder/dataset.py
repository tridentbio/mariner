"""Dataset related classes to use for training/evaluating/testing"""
from collections.abc import Mapping
from typing import TYPE_CHECKING, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData

if TYPE_CHECKING:
    pass


class Collater:
    """
    Collater that automatically handles all of the data types supported by Mariner

    The Collater automatically detects the data types from each element of the batch
    and adjusts the collation function accoringly.

    Args:
        pyg_batch_kwargs - Keyword arguments passed to the PyTorch geometric batch

    Returns:
        Batched data
    """

    def __init__(self, **pyg_batch_kwargs):
        self.pyg_batch_kwargs = pyg_batch_kwargs

    def __call__(self, batch):
        return self.collate(batch)

    def collate(self, batch):  # Deprecated...
        """Prepares batch for layers according to data type

        Use __call__ instead

        Args:
            batch: input with batched data

        Raises:
            TypeError: When can't make a batch out of input
        """
        # Get the first element to check data type
        elem = batch[0]

        # Handle PyG data
        if isinstance(elem, BaseData):
            return Batch.from_data_list(batch, **self.pyg_batch_kwargs)

        # Handle Tensor data
        elif isinstance(elem, torch.Tensor):
            if elem.dtype == torch.long and not all(
                [
                    batch[0].shape == batch[i].shape
                    for i in range(1, len(batch))
                ]
            ):
                return pad_sequence(batch, batch_first=True)
            return default_collate(batch)

        # Handle float data
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)

        # Handle int data
        elif isinstance(elem, int):
            return torch.tensor(batch)

        # Handle str data
        elif isinstance(elem, str):
            return batch

        # Handle Mapping data
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}

        # Handle additional PyG-specific batching
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return type(elem)(*(self(s) for s in zip(*batch)))

        # Handle sequences
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: {type(elem)}")
