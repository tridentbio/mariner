from typing import Any, List
from torch.utils.data import Dataset as TorchDataset
from app.builder.utils import size_repr
from app.features.model.schema.configs import DatasetConfig
from app.features.model.schema.layers_schema import AppmoleculefeaturizerLayerConfig
from torch_geometric.loader import DataLoader
from .storage import BaseStorage
import torch
from torch.utils.data import random_split
import pandas as pd
import pytorch_lightning as pl
import numpy as np


class DataInstance(BaseStorage):
    
    def __init__(self, y = None, **kwargs):

        self.__dict__['_store'] =  BaseStorage(_parent=self)

        if y is not None:
            self.y = y

        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def __getitem__(self, key: str) -> Any:
        return self._store[key]        
        
    def __setitem__(self, key: str, value: Any) -> None:
        self._store[key] = value

    def __delitem__(self, key: str) -> None:
        if key in self._store:
            del self._store[key]
        
    def __getattr__(self, key: str) -> Any:
        if '_store' not in self.__dict__:
            raise RuntimeError
        return getattr(self._store, key)
    
    def __setattr__(self, key: str, value: Any) -> None:
        setattr(self._store, key, value)
        
    def __delattr__(self, key: str, value: Any) -> None:
        delattr(self._store, key)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = [size_repr(k, v, indent=2) for k, v in self._store.items()]
        info = ",\n".join(info)
        return f"{cls}(\n{info}\n)"


class CustomDataset(TorchDataset):
    
    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns,
        featurizers_config,
        target: str = None
    ) -> None:
        self.data = data
        self.target = target
        self.columns = feature_columns
        self._featurizers_config = featurizers_config

        self.setup()

    def _determine_task_type(self) -> str:
        target_type = self.data.dtypes[self.target]
        
        if 'float' in target_type.name:
            # If it is a float target, we treat the task as a regression
            return 'regression'
        
        return AttributeError('Unsupported target type for prediction.')
        
    def setup(self):
        # First we need to determine the type of the task
        if self.target:
            self._task_type = self._determine_task_type()
        # After that we can instanciate all of the featurizers used to transform
        # the columns
        self._featurizers = {}
        for featurizer_config in self._featurizers_config:
            self._featurizers[featurizer_config.name] = featurizer_config.create()
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> DataInstance:
        d = DataInstance()
        sample = dict(self.data.iloc[index, :])
        
        columns_to_include = self.columns.copy()
        # We need to featurize all of the columns that pass into a 
        # featurizer before include in the data instance
        for featurizer in self._featurizers_config:
            d[featurizer.name] = self._featurizers[featurizer.name](sample[featurizer.input[0]])
            columns_to_include.remove(featurizer.input[0])

        # After that we can include all of the columns that remains from the featurizers
        for column in columns_to_include:
            d[column] = sample[column]

        if self.target:
            d.y = sample[self.target]

        return d


class DataModule(pl.LightningDataModule):

    def __init__(self,
        data: pd.DataFrame,
        split_type: str,
        split_target: str,
        batch_size: int,
        dataset_config: DatasetConfig,
        featurizers_config: List[AppmoleculefeaturizerLayerConfig]
    ):
        self.prepare_data_per_node = False
        self.dataset_config = dataset_config
        self.featurizers_config = featurizers_config

        self.data = data

        self.batch_size = batch_size

        self.split_type = split_type
        self.split_target = split_target

        self.dataset = CustomDataset(
            self.data,
            self.dataset_config.feature_columns,
            self.featurizers_config,
            self.dataset_config.target_column
        )

    def setup(self, stage):
        train_split, val_split, _ = self.split_target.split("-")

        full_size = len(self.dataset)
        train_size = int((int(train_split) / 100) * full_size)
        val_size = int((int(val_split) / 100) * full_size)
        test_size = full_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            self.dataset,
            [train_size, val_size, test_size]
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            shuffle=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            shuffle=True
        )
