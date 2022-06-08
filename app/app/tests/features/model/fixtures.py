from typing import Optional

import pandas as pd
import torch
from pandas.core.frame import DataFrame
from torch.utils.data import Dataset

from app.features.model.featurizers import MoleculeFeaturizer


class ExampleDataset(Dataset):
    def __init__(self, path: str, target: Optional[str] = None, **kwargs):
        self.feat = MoleculeFeaturizer(**kwargs)
        self.data: DataFrame = DataFrame(pd.read_csv(path))
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = dict(self.data.iloc[idx, :])
        graph = self.feat(sample["smiles"])

        if self.target:
            graph.y = torch.tensor([sample[self.target]])

        return graph


