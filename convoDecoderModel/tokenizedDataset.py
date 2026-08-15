from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

class TokenDataset(Dataset):
    """
    class definition for tokenized dataset
    """

    path: str | Path
    context_length: int
    dtype: np.uint16

    def __init__(
        self,
        path: str | Path,
        context_length: int,
        dtype=np.uint16
    ):
        self.path = Path(path)
        self.context_length = context_length
        self.tokens = np.memmap(
            self.path,
            dtype=dtype,
            mode='r'
        )
        self.num_sequences = (len(self.tokens) - 1) // self.context_length

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        start = index * self.context_length
        end = start + self.context_length + 1

        tokens = np.array(
            self.tokens[start:end],
            dtype=np.int64,
            copy=True
        )
        tokens = torch.from_numpy(tokens)
        input_ids = tokens[:-1]
        targets = tokens[1:]

        return input_ids, targets

