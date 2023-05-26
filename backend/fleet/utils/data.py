from typing import Dict, Literal, Iterable, Tuple

from humps import camel

from fleet.dataset_schemas import DatasetConfig
from fleet.model_builder.layers_schema import FeaturizersArgsType


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
) -> Tuple[Iterable[FeaturizersArgsType], Iterable[TransformerNormalizer]]:
    """Get's the preprocessing pipeline steps in their topological order.

    Uses the ``forward_args`` from ``dataset_config.featurizers`` and ``dataset_config.transformer``
    attributes to get the dependencies of a preprocessing step.

    Args:
        dataset_config(DatasetConfig): The dataset configuration.

    Returns:
        (featurizers, transforms) tuple, the preprocessing step objects.
    """


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
            featurized = feat_or_transform(value)
        elif isinstance(feat_or_transform, sklearn.base.TransformerMixin):
            featurized = feat_or_transform.fit_transform(value)
        raise RuntimeError()

    for feat in feats:
        key, value = get_args(data, feat)
        apply(feat, value)
        del data[key]  # a column gets only through one pipeline, therefore we can
        # erase the last step

    for transform in transforms:
        key, value = get_args(data, feat)
        apply(transform, value)
        del data[key]

    return data


def build_torch_dataset():
    ...


def build_matrix_dataset():
    ...
