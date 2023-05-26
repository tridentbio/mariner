"""
Utilities to build dataset.
"""
from typing import Dict, Iterable, Literal, Tuple

import networkx as nx
import numpy as np
import sklearn.base
import torch.nn
from humps import camel

from fleet.dataset_schemas import DatasetConfig
from fleet.model_builder.layers_schema import FeaturizersType
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

    class Config:
        """Configures the wrapper class to work as intended."""

        alias_generator = camel.case
        allow_population_by_field_name = True
        allow_population_by_alias = True
        underscore_attrs_are_private = True


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
    featurizers_dict = [feat.dict() for feat in dataset_config.featurizers]
    transforms_dict = [transform.dict() for transform in dataset_config.transforms]

    featurizers_by_name = {feat.name: feat for feat in dataset_config.featurizers}
    transforms_by_name = {
        transform.name: transform for transform in dataset_config.transforms
    }

    graph = make_graph_from_forward_args(featurizers_dict + transforms_dict)
    topo_sort = nx.topological_sort(graph)
    featurizers, transforms = [], []

    # TODO: check that featurizers come before transforms
    for item in topo_sort:
        if item in featurizers_dict:
            featurizers.append(featurizers_by_name[item])
        elif item in transforms_dict:
            transforms.append(transforms_by_name[item])

    return featurizers, transforms


def build_columns_numpy(dataset_config: DatasetConfig):
    # This functions aims to create a dictionary of
    # featurized and preprocessed numpy columns.

    # It can be later used by a dataset, converting it to tensors,
    # Or it can be used by other classes to create a matrix.

    # Inputs:
    # Starting procedure

    # Will store numpy columns as values.
    data = {}

    # This functions raises an error in case some featurizer is
    # applied after a transform, .e.g: col -> transform-> feat
    # would raise errors.
    feats, transforms = dataset_topo_sort(dataset_config)

    def apply(feat_or_transform, numpy_col):
        if isinstance(feat_or_transform, torch.nn.Module):
            feat_or_transform(value)
        elif isinstance(feat_or_transform, sklearn.base.TransformerMixin):
            feat_or_transform.fit_transform(value)
        raise RuntimeError()

    def get_args(data: dict, feat: FeaturizersType) -> np.ndarray:
        """
        Get the parameters necessary to pass to feat by processing it's forward_args and looking
        previously computed values in ``data``
        """
        ...

    for feat in feats:
        value = get_args(data, feat)
        np_array = apply(feat, value)
        data[feat.name] = np_array
        # erase the last step

    for transform in transforms:
        key, value = get_args(data, feat)
        np_array = apply(transform, value)
        data[transform.name] = np_array

    return data


def build_torch_dataset():
    ...


def build_matrix_dataset():
    ...
