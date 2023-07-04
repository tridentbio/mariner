"""Utilities to build datasets."""
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Protocol,
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
from lightning.pytorch.trainer.trainer import TrainerFn
from pydantic import BaseModel
from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset

import fleet.exceptions
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
from fleet.model_builder.featurizers.small_molecule_featurizer import (
    MoleculeFeaturizer,
)
from fleet.model_builder.layers_schema import FeaturizersType
from fleet.model_builder.splitters import apply_split_indexes
from fleet.model_builder.utils import DataInstance, get_references_dict
from fleet.utils.graph import get_leaf_nodes, make_graph_from_forward_args


def make_graph_from_dataset_config(
    dataset_config: DatasetConfig,
) -> nx.DiGraph:
    """Creates a graph from a dataset configuration.

    Args:
        dataset_config(DatasetConfig): The dataset configuration.

    Returns:
        The graph.
    """
    featurizers_dict = {
        feat.name: feat.dict(by_alias=True)
        for feat in dataset_config.featurizers
    }
    transforms_dict = {
        transform.name: transform.dict(by_alias=True)
        for transform in dataset_config.transforms
    }
    preprocessing_steps = list(featurizers_dict.values()) + list(
        transforms_dict.values()
    )
    graph = make_graph_from_forward_args(
        preprocessing_steps + [col.dict() for col in dataset_config.columns]
    )
    for col in dataset_config.columns:
        if graph.has_node(col.name):
            graph.add_node(col.name)

    return graph


def dataset_topo_sort(
    dataset_config: DatasetConfig,
) -> Tuple[Iterable[FeaturizersType], Iterable[Any]]:
    """Get's the preprocessing pipeline steps in their topological order.

    Uses the ``forward_args`` from ``dataset_config.featurizers`` and
    ``dataset_config.transformer`` attributes to get the dependencies of
    a preprocessing step.

    Args:
        dataset_config(DatasetConfig): The dataset configuration.

    Returns:
        (featurizers, transforms) tuple, the preprocessing step objects.
    """
    featurizers_by_name = {
        feat.name: feat for feat in dataset_config.featurizers
    }
    transforms_by_name = {
        transform.name: transform for transform in dataset_config.transforms
    }
    featurizers, transforms = [], []
    graph = make_graph_from_dataset_config(dataset_config)
    topo_sort = nx.topological_sort(graph)
    featurizers_names = [feat.name for feat in dataset_config.featurizers]
    transformers_names = [
        transformer.name for transformer in dataset_config.transforms
    ]
    # TODO: check that featurizers come before transforms
    for item in topo_sort:
        if item in featurizers_names:
            featurizers.append(featurizers_by_name[item])
        elif item in transformers_names:
            transforms.append(transforms_by_name[item])

    return featurizers, transforms


def apply(feat_or_transform, numpy_col):
    """Applies a featurizer or transform to a numpy column."""
    if isinstance(feat_or_transform, torch.nn.Module):
        return feat_or_transform(**numpy_col)
    elif isinstance(feat_or_transform, sklearn.base.TransformerMixin):
        return feat_or_transform.fit_transform(*numpy_col)
    elif callable(feat_or_transform):
        numpy_col = numpy_col[0] if len(numpy_col) == 1 else numpy_col
        arr = []
        for item in numpy_col:
            featurized_item = feat_or_transform(item)
            arr.append(featurized_item)
        if isinstance(feat_or_transform, MoleculeFeaturizer):
            return arr
        arr = np.array(arr)
        return np.array(arr)
    raise TypeError(
        f"feat_or_transform should be a torch.nn.Module or a callable, \
                got {type(feat_or_transform)}"
    )


def has_default_featurizer(column: ColumnConfig):
    """Checks if the column has a default featurizer.

    Args:
        dataset_config(DatasetConfig): The dataset configuration.
        column(ColumnConfig): The column configuration.
    """
    if column.data_type is None:
        return False
    if isinstance(column.data_type, data_types.CategoricalDataType):
        return True
    if isinstance(column.data_type, data_types.DNADataType):
        return True
    if isinstance(column.data_type, data_types.RNADataType):
        return True
    if isinstance(column.data_type, data_types.ProteinDataType):
        return True
    return False


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
        A featurizer callable or None if there is no default featurizer for the
        data type.
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


class ForwardProtocol(Protocol):
    """
    Interface for forward_args attribute of featurizers, transforms and layers.
    """

    forward_args: Union[List[str], Dict[str, Union[List[str], str]], BaseModel]


def cast_np_array(
    data: Union[pd.DataFrame, pd.Series, np.ndarray, Any]
) -> np.ndarray:
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.to_numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)


def get_args(
    data: pd.DataFrame, feat: ForwardProtocol
) -> Union[List[np.ndarray], np.ndarray]:
    """
    Get the parameters necessary to pass to feat by processing it's
    forward_args and looking previously computed values in ``data``.

    Args
        data: The dataframe that holds the dataset data.
        feat: The featurizer or transform.
    """
    if isinstance(feat.forward_args, BaseModel):
        forward_args = feat.forward_args.dict()
    else:
        forward_args = feat.forward_args

    references = get_references_dict(forward_args)

    if isinstance(references, dict):
        result = []
        for value in references.values():
            if not isinstance(value, list):
                value = [value]

            result = [cast_np_array(data[col_name]) for col_name in value]

        return result

    else:
        return np.array([data[col_name] for col_name in references])


TransformFunction = Dict[
    str,
    Tuple[
        Union[TransformerMixin, FunctionTransformer],
        Callable[[Callable, list], Union[np.ndarray, List[np.ndarray]]],
    ],
]


def prepare_data(
    func: Callable[
        [Dict[str, Union[np.ndarray, List[np.ndarray], pd.Series]]],
        Dict[str, Union[np.ndarray, List[np.ndarray], pd.Series]],
    ]
):
    """Decorator for preprocessing the params data.

    Wraps the fit, transform and fit_tranform functions.
    Prepare X and Y to be passed to the modules.
    Apply default featurizers to X and Y if featurize_data_types is True.
    Prepare the dictionary with data to be passed to the func.
    Serialize the result after func is called.

    Args:
        func: fit, fit_transform or transform method from :class:`PreprocessingPipeline`.

    Returns:
        The wrapped function.
    """

    def wrapper(
        self: "PreprocessingPipeline",
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray, None] = None,
    ):
        X, y = self._prepare_X_and_y(X, y)  # pylint: disable=W0212

        if self.featurize_data_types:
            self._apply_default_featurizers(  # pylint: disable=W0212
                X, self.dataset_config.feature_columns
            )

            self._apply_default_featurizers(  # pylint: disable=W0212
                y, self.dataset_config.target_columns
            )

        data = self._dataframe_to_dict(  # pylint: disable=W0212
            pd.concat([X, y], axis=1)
        )
        data = func(self, data)
        return data

    return wrapper


class PreprocessingPipeline:
    """Preprocesses the dataset.

    Args:
        dataset_config: The object describing the columns and data_types of the
            dataset.
    """

    featurizers: TransformFunction
    transforms: TransformFunction

    def __init__(
        self, dataset_config: DatasetConfig, featurize_data_types: bool = True
    ):
        """Creates the pipeline steps without executing them.

        Args:
            dataset_config: The object describing the pipeline.
        """
        self.featurize_data_types = featurize_data_types
        self.dataset_config = dataset_config
        self.featurizers = []
        self.transforms = []
        # Flags the output columns names in the final dataframe
        # It's updated when applying the featurizers/transforms
        graph = make_graph_from_dataset_config(dataset_config)
        self.output_columns = get_leaf_nodes(graph)
        self.featurizers_config = {
            feat.name: feat for feat in dataset_config.featurizers
        }
        self.transforms_config = {
            transform.name: transform
            for transform in dataset_config.transforms
        }
        self.featurizers = {
            feat.name: (
                self._prepare_transform(feat.create()),
                feat.adapt_args_and_apply,
            )
            for feat in dataset_config.featurizers
        }
        self.transforms = {
            transform.name: (
                self._prepare_transform(transform.create()),
                transform.adapt_args_and_apply,
            )
            for transform in dataset_config.transforms
        }
        self._fitted = False

    _state_attrs = [
        "featurizers",
        "transforms",
        "dataset_config",
        "_fitted",
        "output_columns",
    ]

    def get_state(self):
        """
        Returns self attributes needed to be able to load the instance.
        """
        state = {}
        for state_attr in self._state_attrs:
            state[state_attr] = getattr(self, state_attr)
        return state

    @classmethod
    def load_from_state(cls, state: dict):
        """
        Loads an instance from state.
        """
        instance = cls(state["dataset_config"])
        for state_attr in cls._state_attrs:
            setattr(instance, state_attr, state[state_attr])
        return instance

    def _prepare_transform(self, func: Any):
        if isinstance(func, sklearn.base.TransformerMixin):
            return func
        elif callable(func):
            return FunctionTransformer(func)
        else:
            raise ValueError(
                "func must be one of %r"
                % (["sklearn.base.TransformerMixin", "Callable"])
            )

    def get_X_and_y(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits a dataframe into X and y using the dataset_config.

        Args:
            df: The data to be split.

        Returns:
            X and y dataframes.
        """
        feature_columns = [
            col.name for col in self.dataset_config.feature_columns
        ]
        target_columns = [
            col.name for col in self.dataset_config.target_columns
        ]
        return df.loc[:, feature_columns], df.loc[:, target_columns]

    def _prepare_X_and_y(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[None, pd.DataFrame, np.ndarray],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(
                    X,
                    columns=[
                        col.name for col in self.dataset_config.feature_columns
                    ],
                )

            if not isinstance(y, pd.DataFrame):
                y = pd.DataFrame(
                    y,
                    columns=[
                        col.name for col in self.dataset_config.target_columns
                    ],
                )
            return X, y
        except Exception as exc:
            raise TypeError(
                "X and y must be pandas.DataFrame or numpy.ndarray"
            ) from exc

    def _apply_default_featurizers(
        self, data: pd.DataFrame, columns: list[ColumnConfig]
    ):
        for column in columns:
            feat = get_default_data_type_featurizer(
                self.dataset_config, column
            )
            if feat is not None:
                feat = self._prepare_transform(feat)
                value = data[column.name]
                data.drop(labels=[column.name], axis="columns", inplace=True)
                data[column.name] = feat.transform(value)

    def get_preprocess_steps(self):
        """
        Returns the featurizers and transforms in the order they are applied.
        """
        feats, transforms = map(list, dataset_topo_sort(self.dataset_config))
        for config in feats:
            yield (config, self.featurizers[config.name])
        for config in transforms:
            yield (config, self.transforms[config.name])

    def _dataframe_to_dict(
        self, df
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray], pd.Series]]:
        dictionary = {}
        for column in df.columns:
            dictionary[column] = df[column]
        return dictionary

    @prepare_data
    def fit(
        self, data: Dict[str, Union[np.ndarray, List[np.ndarray], pd.Series]]
    ):
        """
        Fits the featurizers and transforms to the data.
        """
        for config, (
            transformer,
            adapt_args_and_apply,
        ) in self.get_preprocess_steps():
            args = get_args(data, config)
            try:
                adapt_args_and_apply(transformer.fit, args)
            except Exception as exp:
                raise fleet.exceptions.FitError(
                    f"Failed to fit {config}"
                ) from exp

        self._fitted = True

    @prepare_data
    def transform(
        self, data: Dict[str, Union[np.ndarray, List[np.ndarray], pd.Series]]
    ):
        """Transforms X and y according to preprocessor config.

        Args:
            X: Data matrix.
            y: Target column matrix.

        Raises:
            fleet.exceptions.Transform: When the preprocessing steps have not
                been fitted.
        """
        for config, (
            transformer,
            adapt_args_and_apply,
        ) in self.get_preprocess_steps():
            args = get_args(data, config)
            try:
                transformed = adapt_args_and_apply(transformer.transform, args)
                data[config.name] = transformed
            except Exception as exp:
                raise fleet.exceptions.TransformError(
                    f"Failed to transform {config}"
                ) from exp

        self._fitted = True
        return data

    @prepare_data
    def fit_transform(
        self, data: Dict[str, Union[np.ndarray, List[np.ndarray], pd.Series]]
    ):
        """Fits and transforms X and y according to preprocessor config.

        Args:
            X: Data matrix.
            y: Target column matrix.

        Raises:
            fleet.exceptions.FitError: When it fails to fit
        """
        for config, (
            transformer,
            adapt_args_and_apply,
        ) in self.get_preprocess_steps():
            args = get_args(data, config)
            try:
                result = adapt_args_and_apply(transformer.fit_transform, args)
                data[config.name] = result
            except Exception as exp:
                raise fleet.exceptions.FitError(
                    f"Failed to fit {config}"
                ) from exp
        self._fitted = True

        return data


def adapt_numpy_to_tensor(
    arr: Union[list[np.ndarray], np.ndarray]
) -> Union[list[np.ndarray], np.ndarray, torch.Tensor]:
    """Creates a tensor with the same shape and type as the numpy array.

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
        tensor = torch.as_tensor(arr).view((-1,))
        tensor = tensor.long()
        return tensor
    elif arr_type == "float":
        tensor = torch.as_tensor(arr).view((-1,))
        tensor = tensor.float()
        return tensor

    return arr


class MarinerTorchDataset(Dataset):
    """The dataset class that holds the data and the preprocessing steps.

    Attributes:
        dataset_config: The object describing the columns and data_types of
            the dataset.
        model_config: The object describing the model architecture and
            hyperparameters.
        data: The data to be used by the model.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        dataset_config: DatasetConfig,
        preprocessing_pipeline: Union[None, PreprocessingPipeline] = None,
    ):
        """MarinerTorchDataset constructor.

        Args:
            data: The data to be used by the model.
            dataset_config: The object describing the columns and data_types
                of the dataset.
            model_config: The object describing the model architecture and
                hyperparameters.
        """
        self.data = data
        if preprocessing_pipeline is None:
            self.preprocessing_pipeline = PreprocessingPipeline(dataset_config)
        else:
            self.preprocessing_pipeline = preprocessing_pipeline
        step = self.data["step"]
        transformed = self.preprocessing_pipeline.transform(
            *self.preprocessing_pipeline.get_X_and_y(self.data)
        )
        for field, result in transformed.items():
            self.data[field] = result

        self.data["step"] = step

    def get_subset_idx(self, step=None) -> list[int]:
        """Gets the indices of where step equals the step column in the
        dataset.

        Returned indices can be used to instantiate a new dataset with the
        subset of data.

        Args:
            step: The step to get the indices for.

        Returns:
            The indices of where step equals the step column in the dataset.
        """
        if step is None:
            return list(range(len(self.data)))
        return (self.data[self.data["step"] == step]).index.tolist()

    def __getitem__(self, idx: int):
        """Gets a sample from the dataset.

        Args:
            idx: The index of the sample to get.

        Returns:
            The sample at the given index.
        """
        data = DataInstance()

        output_columns = self.preprocessing_pipeline.output_columns

        for column in self.data[output_columns]:
            data[column] = adapt_numpy_to_tensor(self.data.loc[idx, column])

        return data

    def __len__(self):
        return len(self.data)


class DataModule(pl.LightningDataModule):
    """DataModule is responsible for integrating and handling the dataloaders
    of each step of an experiment (training, testing and validation) in order
    to provide a pytorch lightning compatible interface to speed up and
    facilitate its maintenance.

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

    train_dataset: Subset[Any] = None
    val_dataset: Subset[Any] = None
    test_dataset: Subset[Any] = None

    def __init__(
        self,
        data: pd.DataFrame,
        config: DatasetConfig,
        batch_size=32,
        split_type: Union[None, Literal["random", "scaffold"]] = None,
        split_target: Union[None, str] = None,
        split_column: Union[None, str] = None,
        collate_fn: Union[Callable, None] = None,
        preprocessing_pipeline: Union[None, PreprocessingPipeline] = None,
    ):
        super().__init__()
        self.dataset_config = config
        self.featurizers_config = config.featurizers
        self.data = data
        self.batch_size = batch_size
        self.split_type = split_type
        self.split_target = split_target
        self.split_column = split_column
        if preprocessing_pipeline is None:
            self.preprocessing_pipeline = PreprocessingPipeline(
                self.dataset_config
            )
        else:
            self.preprocessing_pipeline = preprocessing_pipeline

        if collate_fn is None:
            self.collate_fn = Collater()
        else:
            self.collate_fn = collate_fn

    def setup(self, stage: TrainerFn):
        """Method used for pytorch lightning to setup the data module instance
        and prepare the data loaders to be used in the trainer.

        Args:
            stage (_type_, optional): _description_. Defaults to None.
        """
        if "step" not in self.data.columns:
            apply_split_indexes(
                df=self.data,
                split_type=self.split_type or "random",
                split_target=self.split_target or "60-20-20",
                split_column=self.split_column,
            )

        self.preprocessing_pipeline.fit(
            self.data["step"] == TrainingStep.TRAIN.value
        )
        dataset = MarinerTorchDataset(
            data=self.data,
            dataset_config=self.dataset_config,
        )
        if stage == TrainerFn.FITTING:
            self.train_dataset = Subset(
                dataset=dataset,
                indices=dataset.get_subset_idx(TrainingStep.TRAIN.value),
            )
        if stage in [TrainerFn.FITTING, TrainerFn.VALIDATING]:
            self.val_dataset = Subset(
                dataset=dataset,
                indices=dataset.get_subset_idx(TrainingStep.VAL.value),
            )
        if stage in [TrainerFn.FITTING, TrainerFn.TESTING]:
            self.test_dataset = Subset(
                dataset=dataset,
                indices=dataset.get_subset_idx(TrainingStep.TEST.value),
            )

    def train_dataloader(self) -> DataLoader:
        """Return the DataLoader instance used to train the custom model.

        Returns:
            DataLoader: instance used in train steps.
        """
        assert self.train_dataset, 'setup(stage="fit") must be called'
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the DataLoader instance used to test the custom model.

        Returns:
            DataLoader: instance used in test steps.
        """

        assert self.test_dataset, 'setup(stage="test") must be called'
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
        assert self.val_dataset, 'setup(stage="validate") must be called'
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            collate_fn=self.collate_fn,
        )
