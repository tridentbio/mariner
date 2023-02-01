"""Dataset related classes to use for training/evaluating/testing"""
from typing import Any, Callable, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torch_geometric.data import Dataset as PygDataset

from model_builder.component_builder import AutoBuilder
from model_builder.featurizers.base_featurizers import BaseFeaturizer
from model_builder.featurizers.integer_featurizer import IntegerFeaturizer
from model_builder.model_schema_query import (
    get_dependencies,
    get_target_columns,
)
from model_builder.schemas import (
    CategoricalDataType,
    ColumnConfig,
    DNADataType,
    ModelSchema,
    NumericalDataType,
    ProteinDataType,
    QuantityDataType,
    RNADataType,
)
from model_builder.utils import DataInstance, get_references_dict


class CustomDataset(PygDataset):
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
            Object containing information about the fAcho que vaeaturizers
            used by the CustomDataset.
        target (str, optional): Name of the columns that will be
            used as the model target for predictions.

    Example:
    >>> dataset = CustomDataset(data, ['mwt', 'smiles'], featurizer_config, 'tpsa')
    """

    def __init__(self, data: pd.DataFrame, config: ModelSchema, target=True) -> None:
        super().__init__()
        self.data = data
        self.config = config
        self.target = target
        self.setup()

    def get_featurizer_configs(self):
        """Gets the input featurizers"""
        return self.config.featurizers

    def setup(self):
        """Instantiates input and output featurizers"""
        # Instantiate all featurizers
        self._featurizers = {}
        for featurizer_config in self.get_featurizer_configs():
            feat = featurizer_config.create()
            if isinstance(feat, AutoBuilder):
                feat.set_from_model_schema(
                    self.config, list(get_dependencies(featurizer_config))
                )
            self._featurizers[featurizer_config.name] = feat
        self.output_featurizer = self.get_output_featurizer()

    def _get_default_featurizer(
        self, column: ColumnConfig
    ) -> Union[BaseFeaturizer, None]:
        """Gets a default featurizer based on the data type"""
        feat = None
        if isinstance(column.data_type, CategoricalDataType):
            feat = IntegerFeaturizer()
            feat.set_from_model_schema(self.config, [column.name])
        return feat

    def get_output_featurizer(self) -> Union[BaseFeaturizer, None]:
        """Gets the output featurizer"""
        if self.target:
            targets = get_target_columns(self.config)
            # Assume a single target
            target = targets[0]
            return self._get_default_featurizer(target)


    def __len__(self) -> int:
        """Gets the number of rows in the dataset"""
        return len(self.data)

    def __getitem__(self, index) -> DataInstance:
        """Gets the item at index-th row of the dataset

        Args:
            index (int): row to get
        Returns:
            DataInstance with all values of that row
        """
        # Instantiate the data instance
        data = DataInstance()
        
        # Convert the row to a dictionary
        sample = dict(self.data.iloc[index, :])
        
        # Subset columns
        columns_to_include = self.config.dataset.feature_columns
        
        # Featurize all of the columns that pass into a
        # featurizer before including in the data instance
        for featurizer in self.get_featurizer_configs():
            
            references = get_references_dict(featurizer.forward_args.dict())
            assert len(references) == 1, "only 1 forward arg for featurizers for now"
            col_name = list(references.values())[0]
            data[featurizer.name] = self._featurizers[featurizer.name](sample[col_name])

            # Remove featurized columns from columns_to_include
            # since its featurized value was already included
            for index, col in enumerate(columns_to_include):
                if col.name == col_name:
                    columns_to_include.pop(index)

        # Include all unfeaturized columns
        for column in columns_to_include:
            val = sample[column.name]
            if isinstance(column.data_type, (NumericalDataType, QuantityDataType)):
                data[column.name] = torch.Tensor([val])
            elif isinstance(
                column.data_type, (DNADataType, RNADataType, ProteinFeaturizer)
            ):
                feat = self._get_default_featurizer(column)
                assert feat, "dna, rna and protein have a default featurizer"
                data[column.name] = feat(val)
            else:
                data[column.name] = val

        if self.target:
            targets = get_target_columns(self.config)
            target = targets[0]
            if self.output_featurizer:
                data.y = self.output_featurizer(sample[target.name])
            else:
                data.y = torch.Tensor([sample[target.name]])

        return data


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
        collate_fn (Callable, optional): Function to be called on a list
            of samples by the dataloader to build a batch.
    """

    train_dataset: Subset[Any]
    val_dataset: Subset[Any]
    test_dataset: Subset[Any]

    def __init__(
        self,
        data: pd.DataFrame,
        split_type: str,
        split_target: str,
        config: ModelSchema,
        batch_size=32,
        collate_fn: Union[Callable, None] = None,
    ):
        print("Creating a DataModule")
        super().__init__()
        self.dataset_config = config.dataset
        self.featurizers_config = config.featurizers

        self.data = data

        self.batch_size = batch_size

        self.split_type = split_type
        self.split_target = split_target

        self.collate_fn = collate_fn
        self.dataset = CustomDataset(
            self.data,
            config,
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

        # TODO: Use split index column instead
        train_split, val_split, _ = self.split_target.split("-")

        full_size = len(self.dataset)
        train_size = int((int(train_split) / 100) * full_size)
        val_size = int((int(val_split) / 100) * full_size)
        test_size = full_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )

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
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the DataLoader instance used to test the custom
        model.

        Returns:
            DataLoader: instance used in test steps.
        """
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the DataLoader used to validate the custom model.

        Returns:
            DataLoader: instance used in validation steps.
        """
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            collate_fn=self.collate_fn,
        )
