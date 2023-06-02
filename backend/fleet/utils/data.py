"""
Utilities to build dataset.
"""
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    Iterable,
    Literal,
    Tuple,
    Union,
)

import lightning.pytorch as pl
import networkx as nx
import numpy as np
import pandas as pd
import sklearn.base
import torch
import torch.nn
from humps import camel
from pydantic import Field
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset

from fleet import data_types
from fleet.dataset_schemas import ColumnConfig, DatasetConfig
from fleet.model_builder.constants import TrainingStep
from fleet.model_builder.dataset import Collater
from fleet.model_builder.featurizers import (
    DNASequenceFeaturizer,
    IntegerFeaturizer,
    ProteinSequenceFeaturizer,
    RNASequenceFeaturizer,
)
from fleet.model_builder.featurizers.base_featurizers import BaseFeaturizer
from fleet.model_builder.featurizers.small_molecule_featurizer import MoleculeFeaturizer
from fleet.model_builder.layers_schema import FeaturizersType
from fleet.model_builder.schemas import TorchModelSchema
from fleet.model_builder.utils import DataInstance, get_references_dict
from fleet.utils.graph import make_graph_from_forward_args


def _get_default_featurizer(
    column: "ColumnConfig",
    config: Union[None, TorchModelSchema] = None,
    dataset_config: Union[None, DatasetConfig] = None,
) -> Union[BaseFeaturizer, None]:
    """Gets a default featurizer based on the data type"""
    feat = None
    if isinstance(column.data_type, data_types.CategoricalDataType):
        feat = IntegerFeaturizer()
        feat.set_from_model_schema(
            config=config,
            deps=[column.name],
            dataset_config=dataset_config,
        )
    elif isinstance(column.data_type, data_types.DNADataType):
        feat = DNASequenceFeaturizer()
    elif isinstance(column.data_type, data_types.RNADataType):
        feat = RNASequenceFeaturizer()
    elif isinstance(column.data_type, data_types.ProteinDataType):
        feat = ProteinSequenceFeaturizer()

    return feat


class TransformerNormalizer:
    """
    Temporary class
    """

    name: str
    type: Literal[
        "sklearn.preprocessing.Normalizer"
    ] = "sklearn.preprocessing.Normalizer"
    forward_args: Dict[str, str]
    constructor_args: Dict[str, str]

    def create(self):
        ...

    class Config:
        """Configures the wrapper class to work as intended."""

        alias_generator = camel.case
        allow_population_by_field_name = True
        allow_population_by_alias = True
        underscore_attrs_are_private = True


FeatOrTransf = Annotated[
    Union[FeaturizersType, TransformerNormalizer], Field(discriminator="type")
]


def dataset_topo_sort(
    dataset_config: DatasetConfig,
) -> Tuple[Iterable[FeaturizersType], Iterable[TransformerNormalizer]]:
    """Get's the preprocessing pipeline steps in their topological order.

    Uses the ``forward_args`` from ``dataset_config.featurizers`` and ``dataset_config.transformer``
    attributes to get the dependencies of a preprocessing step.

    Args:
        dataset_config(DatasetConfig): The dataset configuration.

    Returns:
        (featurizers, transforms) tuple, the preprocessing step objects.
    """
    featurizers_dict = {
        feat.name: feat.dict(by_alias=True) for feat in dataset_config.featurizers
    }
    transforms_dict = {
        transform.name: transform.dict(by_alias=True)
        for transform in dataset_config.transforms
    }

    featurizers_by_name = {feat.name: feat for feat in dataset_config.featurizers}
    transforms_by_name = {
        transform.name: transform for transform in dataset_config.transforms
    }

    preprocessing_steps = list(featurizers_dict.values()) + list(
        transforms_dict.values()
    )
    graph = make_graph_from_forward_args(preprocessing_steps)
    topo_sort = nx.topological_sort(graph)
    featurizers, transforms = [], []

    # TODO: check that featurizers come before transforms
    for item in topo_sort:
        if item in featurizers_dict:
            featurizers.append(featurizers_by_name[item])
        elif item in transforms_dict:
            transforms.append(transforms_by_name[item])

    return featurizers, transforms


def apply(feat_or_transform, numpy_col):
    """
    Applies a featurizer or transform to a numpy column.
    """
    if isinstance(feat_or_transform, torch.nn.Module):
        return feat_or_transform(numpy_col)
    elif isinstance(feat_or_transform, sklearn.base.TransformerMixin):
        return feat_or_transform.fit_transform(numpy_col)
    elif callable(feat_or_transform):
        arr = []
        for item in numpy_col:
            featurized_item = feat_or_transform(item)
            arr.append(featurized_item)
        if isinstance(feat_or_transform, MoleculeFeaturizer):
            return torch.Tensor(arr)
        return np.array(arr)
    raise RuntimeError()


def get_default_data_type_featurizer(
    dataset_config: DatasetConfig, column: "ColumnConfig"
) -> Union[Callable, None]:
    """Gets a default featurizer based on the data type.

    Here is a table with the default featurizers for each data type:

    +--------------------------+--------------------------+
    | Data Type                | Default Featurizer       |
    +==========================+==========================+
    | CategoricalDataType      | IntegerFeaturizer        |
    +--------------------------+--------------------------+
    | DNADataType              | DNASequenceFeaturizer    |
    +--------------------------+--------------------------+
    | RNADataType              | RNASequenceFeaturizer    |
    +--------------------------+--------------------------+
    | ProteinDataType          | ProteinSequenceFeaturizer|
    +--------------------------+--------------------------+

    Args:
        dataset_config(DatasetConfig): The dataset configuration.
        column(ColumnConfig): The column configuration.

    Returns:
        A featurizer callable or None if there is no default featurizer for the data type.
    """
    feat = None
    if isinstance(column.data_type, data_types.CategoricalDataType):
        feat = IntegerFeaturizer()
        feat.set_from_model_schema(
            dataset_config=dataset_config,
            deps=[column.name],
        )
    elif isinstance(column.data_type, data_types.DNADataType):
        feat = DNASequenceFeaturizer()
    elif isinstance(column.data_type, data_types.RNADataType):
        feat = RNASequenceFeaturizer()
    elif isinstance(column.data_type, data_types.ProteinDataType):
        feat = ProteinSequenceFeaturizer()

    return feat


def get_args(data: pd.DataFrame, feat: FeaturizersType) -> np.ndarray:
    """
    Get the parameters necessary to pass to feat by processing it's forward_args and looking
    previously computed values in ``data``
    """
    references = get_references_dict(feat.forward_args.dict())
    assert (
        len(references) == 1
    ), "only 1 forward arg for featurizer/tranform is allowed currently"
    col_name = list(references.values())[0]
    return data[col_name].to_numpy()


def build_columns_numpy(
    dataset_config: DatasetConfig, df: pd.DataFrame, featurize_outputs=True
) -> pd.DataFrame:
    """Updates the dataframe with the preprocessed columns.

    Uses ``forward_args`` of botch featurizers and transforms defined in ``dataset_config``
    to iterate topologically over the preprocessing steps.

    Args:
        dataset_config: The object describing the columns and data_types of the dataset.
        df: The :class:`pd.DataFrame` that holds the dataset data.

    Returns:
        The updated ``DataFrame`` with the preprocessed dataset.
    """
    # Get featurizers and transformers in order
    feats, transforms = map(list, dataset_topo_sort(dataset_config))

    # Add featurized columns into dataframe.
    for feat in feats:
        value = get_args(df, feat)
        df[feat.name] = apply(feat.create(), value)

    # Add transforms into dataframe.
    for transform in transforms:
        value = get_args(df, transform)
        df[transform.name] = apply(transform.create(), value)

    # Apply default featurizers to output columns.
    if featurize_outputs:
        # Add default featurizers into dataframe.
        for column in dataset_config.feature_columns:
            feat = get_default_data_type_featurizer(dataset_config, column)
            if feat is not None:
                df[column.name] = apply(feat, df[column.name])
        for target in dataset_config.target_columns:
            feat = get_default_data_type_featurizer(dataset_config, target)
            if feat is not None:
                df[target.name] = apply(feat, df[target.name])

    return df


def adapt_numpy_to_tensor(arr: Union[list[np.ndarray], np.ndarray]) -> torch.Tensor:
    """
    Creates a tensor with the same shape and type as the numpy array.

    Args:
        arr: The numpy array to be converted to a tensor.

    Returns:
        The tensor with the same shape and type as the numpy array.
    """

    def _get_type(arr):
        if isinstance(arr, (list, np.ndarray)):
            return _get_type(arr[0])
        if isinstance(arr, str):
            return "str"
        if issubclass(type(arr), (np.int64, np.int32)):  # type: ignore
            return "int"
        elif isinstance(arr, (np.float32, np.float64)):  # type: ignore
            return "float"

    arr_type = _get_type(arr)
    if arr_type == "int":
        tensor = torch.as_tensor(arr)
        tensor = tensor.long()
        return tensor
    elif arr_type == "float":
        tensor = torch.as_tensor(arr)
        tensor = tensor.float()
        return tensor
    else:
        print("arr is not a list of numpy floats or ints")
        print("arr is of type: ", type(arr))
        if isinstance(arr, list):
            print("arr[0] is of type: ", type(arr[0]))
    return arr


class MarinerTorchDataset(Dataset):
    """
    The dataset class that holds the data and the preprocessing steps.

    Attributes:
        dataset_config: The object describing the columns and data_types of the dataset.
        model_config: The object describing the model architecture and hyperparameters.
        data: The data to be used by the model.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        dataset_config: DatasetConfig,
    ):
        """MarinerTorchDataset constructor.

        Args:
            data: The data to be used by the model.
            dataset_config: The object describing the columns and data_types of the dataset.
            model_config: The object describing the model architecture and hyperparameters.
        """
        self.dataset_config = dataset_config
        self.data = build_columns_numpy(self.dataset_config, data)

    def get_subset_idx(self, step=None) -> list[int]:
        """
        Gets the indices of where step equals the step column in the dataset.

        Returned indices can be used to instantiate a new dataset with the subset of data.

        Args:
            step: The step to get the indices for.

        Returns:
            The indices of where step equals the step column in the dataset.
        """
        if step is None:
            return list(range(len(self.data)))
        assert len(self.data["step"] == step) == len(self.data), "will break."
        return (self.data[self.data["step"] == step]).index.tolist()

    def __getitem__(self, idx: int):
        """Gets a sample from the dataset.

        Args:
            idx: The index of the sample to get.

        Returns:
            The sample at the given index.
        """
        item = DataInstance()
        sample = dict(self.data.iloc[idx, :])

        for col in self.dataset_config.feature_columns:
            item[col.name] = adapt_numpy_to_tensor([sample[col.name]])
        for col in self.dataset_config.featurizers:
            item[col.name] = adapt_numpy_to_tensor([sample[col.name]])
        for col in self.dataset_config.transforms:
            item[col.name] = adapt_numpy_to_tensor([sample[col.name]])

        for target in self.dataset_config.target_columns:
            item[target.name] = adapt_numpy_to_tensor([sample[target.name]])

        return item

    def __len__(self):
        return len(self.data)


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
        model_config: TorchModelSchema,
        config: DatasetConfig,
        batch_size=32,
        collate_fn: Union[Callable, None] = Collater(),
    ):
        super().__init__()
        self.dataset_config = config
        self.featurizers_config = config.featurizers

        self.data = data

        self.batch_size = batch_size

        self.split_type = split_type
        self.split_target = split_target

        self.collate_fn = collate_fn
        self.dataset = MarinerTorchDataset(data=self.data, dataset_config=config)

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
        self.train_dataset = Subset(
            dataset=self.dataset,
            indices=self.dataset.get_subset_idx(TrainingStep.TRAIN.value),
        )
        self.val_dataset = Subset(
            dataset=self.dataset,
            indices=self.dataset.get_subset_idx(TrainingStep.VAL.value),
        )
        self.test_dataset = Subset(
            dataset=self.dataset,
            indices=self.dataset.get_subset_idx(TrainingStep.TEST.value),
        )

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
            self.val_dataset,
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
