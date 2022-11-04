from typing import List, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from model_builder.layers_schema import FeaturizersType
from model_builder.schemas import ColumnConfig, DatasetConfig
from model_builder.utils import DataInstance, get_references_dict


class CustomDataset(TorchDataset):
    """Class that implements a custom dataset to support multiple
    inputs from multiple different layers.

    The CustomDataset makes use of the DataInstance to allow a kind
    of dynamic storage that works as a hashmap, in this way you can
    store content from different layers to a neural network with more
    than one input.

    See:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
        https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset

    Args:
        data (pd.DataFrame): DataFrame instance with the data that
            will be used by the CustomDataset.
        feature_columns (List[str]): List of columns that will be
            used by the model to be extracted.
        featurizers_config (List[AppmoleculefeaturizerLayerConfig]):
            Object containing information about the featurizers
            used by the CustomDataset.
        target (str, optional): Name of the columns that will be
            used as the model target for predictions.

    Example:
    >>> dataset = CustomDataset(data, ['mwt', 'smiles'], featurizer_config, 'tpsa')
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: List[ColumnConfig],
        featurizers_config,
        target: Optional[ColumnConfig] = None,
    ) -> None:
        super().__init__()
        self.data = data
        self.target = target
        self.columns = feature_columns
        self._featurizers_config = featurizers_config

        self.setup()

    def _determine_task_type(self) -> str:
        """Determine the task type based on the column selected as
        target.

        Returns:
            str: String "regression" or "classification".
        """
        assert self.target, "cannot auto determine task type when target = None"
        target_type = self.data.dtypes[self.target.name]

        if "float" in target_type.name:
            # If it is a float target, we treat the task as a regression
            return "regression"

        raise AttributeError("Unsupported target type for prediction.")

    def setup(self):
        # Determine the type of the task
        if self.target:
            self._task_type = self._determine_task_type()
        # Instanciate all featurizers
        self._featurizers = {}
        for featurizer_config in self._featurizers_config:
            self._featurizers[featurizer_config.name] = featurizer_config.create()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> DataInstance:
        d = DataInstance()
        sample = dict(self.data.iloc[index, :])
        columns_to_include = self.columns.copy()
        # Featurize all of the columns that pass into a
        # featurizer before include in the data instance
        for featurizer in self._featurizers_config:
            references = get_references_dict(featurizer.forward_args.dict())
            assert len(references) == 1, "only 1 forward arg for featurizers for now"
            col_name = list(references.values())[0]
            d[featurizer.name] = [self._featurizers[featurizer.name](sample[col_name])]
            # Remove featurized columns from columsn_to_include
            # since it's featurized value was already included
            for index, col in enumerate(columns_to_include):
                if col.name == col_name:
                    columns_to_include.pop(index)

        # include all unfeaturized columns
        for column in columns_to_include:
            d[column.name] = torch.Tensor([sample[column.name]])

        if self.target:
            d.y = torch.Tensor([sample[self.target.name]])

        assert "MolToGraphFeaturizer" in d
        assert "mwt" in d

        return d


class DataModule(pl.LightningDataModule):
    """DataModule is responsible for integrating and handling the
    dataloaders of each step of an experiment (training, testing and
    validation) in order to provide a pytorch lightning compatible
    interface to speed up and facilitate its maintenance.

    Args:
        data (pd.DataFrame): Pandas DataFrame that contains the
            dataset to be used by the model.
        split_type (str): Type of split that will be performed
            on the dataset, it can be random or scaffold.
        split_target (str): String containing information about
            the split target, for random the split_target will
            contain a string with the split size for the dataset
            in the 3 steps with the following format, training-val-test.
        featurizers_config (List[AppmoleculefeaturizerLayerConfig]):
            Object containing information about the featurizers
            used by the CustomDataset.
        dataset_config (DatasetConfig): Object containing information
            about the Dataset used.
        batch_size (int, optional): Number of data instances in each
            batch. Defaults to 32.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        split_type: str,
        split_target: str,
        featurizers_config: FeaturizersType,
        dataset_config: DatasetConfig,
        batch_size: int = 32,
    ):
        super().__init__()
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
            self.dataset_config.target_column,
        )

    def setup(self, stage=None):
        """Method used for pytorch lightning to setup the data
        module instance and prepare the data loaders to be used in
        the trainer.

        See:
            https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.LightningDataModule.html#pytorch_lightning.core.LightningDataModule
            https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html

        Args:
            stage (_type_, optional): _description_. Defaults to None.
        """
        train_split, val_split, _ = self.split_target.split("-")

        full_size = len(self.dataset)
        train_size = int((int(train_split) / 100) * full_size)
        val_size = int((int(val_split) / 100) * full_size)
        test_size = full_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )

        print(train_dataset, val_dataset, test_dataset)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self) -> DataLoader:
        """Return the DataLoader instance used to train the custom
        model.

        Returns:
            DataLoader: instance used in train steps.
        """
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the DataLoader instance used to test the custom
        model.

        TODO: maybe whe can set drop last or other params in dl.

        Returns:
            DataLoader: instance used in test steps.
        """
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        """Return the DataLoader used to validate the custom model.

        Returns:
            DataLoader: instance used in validation steps.
        """
        return DataLoader(self.test_dataset, self.batch_size, shuffle=True)
