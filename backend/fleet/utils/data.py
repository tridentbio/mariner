"""
Utilities to build dataset.
"""
from typing import Annotated, Callable, Dict, Iterable, Literal, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import sklearn.base
import torch.nn
from humps import camel
from pydantic import Field

from fleet import data_types
from fleet.dataset_schemas import ColumnConfig, DatasetConfig
from fleet.model_builder.featurizers import (
    DNASequenceFeaturizer,
    IntegerFeaturizer,
    ProteinSequenceFeaturizer,
    RNASequenceFeaturizer,
)
from fleet.model_builder.layers_schema import FeaturizersType
from fleet.model_builder.utils import get_references_dict
from fleet.utils.graph import make_graph_from_forward_args


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

    print("%r" % topo_sort)
    print("featurizers %r" % [f for f in featurizers_dict])
    print("transforms %r" % [t for t in transforms_dict])

    # TODO: check that featurizers come before transforms
    for item in topo_sort:
        if item in featurizers_dict:
            featurizers.append(featurizers_by_name[item])
        elif item in transforms_dict:
            transforms.append(transforms_by_name[item])

    return featurizers, transforms


def apply(feat_or_transform, numpy_col):
    print(feat_or_transform)
    print(feat_or_transform.__class__)
    if isinstance(feat_or_transform, torch.nn.Module):
        return feat_or_transform(numpy_col)
    elif isinstance(feat_or_transform, sklearn.base.TransformerMixin):
        return feat_or_transform.fit_transform(numpy_col)
    elif callable(feat_or_transform):
        return np.array([feat_or_transform(item) for item in numpy_col])
    raise RuntimeError()


def get_default_data_type_featurizer(
    dataset_config: DatasetConfig, column: "ColumnConfig"
) -> Union[Callable, None]:
    """Gets a default featurizer based on the data type"""
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
    dataset_config: DatasetConfig, df: pd.DataFrame
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

    print("Number of feats: %d" % len(feats))
    print("Number of transforms: %d" % len(transforms))

    # Index preprocessing_steps by name
    preprocessing_steps = {}

    # Fill featurizers and transforms into preprocessing_steps index
    for featurizer_config in feats + transforms:
        feat = featurizer_config.create()
        preprocessing_steps[featurizer_config.name] = feat

    print("%r" % preprocessing_steps)
    # Add featurized columns into dataframe.
    for feat in feats:
        value = get_args(df, feat)
        df[feat.name] = apply(feat.create(), value)

    print("%r" % df.columns)
    # Add default featurizers into dataframe.
    for column in dataset_config.feature_columns:
        if isinstance(
            column.data_type,
            (
                data_types.DNADataType,
                data_types.RNADataType,
                data_types.ProteinDataType,
            ),
        ):
            feat = get_default_data_type_featurizer(dataset_config, column)
            if feat is not None:
                df[column.name] = feat(df[column.name])

    # Add transforms into dataframe.
    for transform in transforms:
        value = get_args(df, transform)
        df[transform.name] = apply(transform.create(), value)

    return df
