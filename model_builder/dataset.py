import pandas as pd
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import random_split
from torch_geometric.data import Dataset as PygDataset
from torch_geometric.loader import DataLoader

from model_builder.component_builder import AutoBuilder
from model_builder.model_schema_query import (
    get_columns_configs,
    get_dependencies,
    get_target_columns,
)
from model_builder.schemas import CategoricalDataType, ModelSchema
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
            Object containing information about the featurizers
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
        return self.config.featurizers

    def setup(self):
        # Instanciate all featurizers
        self._featurizers = {}
        for featurizer_config in self.get_featurizer_configs():
            feat = featurizer_config.create()
            if isinstance(feat, AutoBuilder):
                feat.set_from_model_schema(
                    self.config, list(get_dependencies(featurizer_config))
                )
            self._featurizers[featurizer_config.name] = feat

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> DataInstance:
        d = DataInstance()
        sample = dict(self.data.iloc[index, :])
        columns_to_include = get_columns_configs(self.config).copy()
        # Featurize all of the columns that pass into a
        # featurizer before include in the data instance
        for featurizer in self.get_featurizer_configs():
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
            val = sample[column.name]
            if isinstance(val, (float, int)):
                d[column.name] = torch.Tensor([val])
            else:
                d[column.name] = val

        # TODO: Fix this: d.y should be the featurized output of the target columns
        if self.target:
            targets = get_target_columns(self.config)
            target = targets[0]  # Assume a single target
            if isinstance(target.data_type, CategoricalDataType):
                classes = target.data_type.classes
                idx = classes[sample[target.name]]
                d.y = F.one_hot(torch.tensor(idx), num_classes=len(classes)).float()
            else:
                d.y = torch.Tensor([sample[target.name]])

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
        config: ModelSchema,
        batch_size=32,
    ):
        super().__init__()
        self.dataset_config = config.dataset
        self.featurizers_config = config.featurizers

        self.data = data

        self.batch_size = batch_size

        self.split_type = split_type
        self.split_target = split_target

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
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the DataLoader instance used to test the custom
        model.

        TODO: maybe whe can set drop last or other params in dl.

        Returns:
            DataLoader: instance used in test steps.
        """
        return DataLoader(self.test_dataset, self.batch_size)

    def val_dataloader(self) -> DataLoader:
        """Return the DataLoader used to validate the custom model.

        Returns:
            DataLoader: instance used in validation steps.
        """
        return DataLoader(self.val_dataset, self.batch_size)
