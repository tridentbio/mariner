from typing import List
from app.builder.dataset import DataModule
from app.features.dataset.model import Dataset
from app.features.model.schema.configs import DatasetConfig, ModelConfig
from app.features.model.schema.layers_schema import AppmoleculefeaturizerLayerConfig


def build_run(configs):
    pass


def build_model(model_config: ModelConfig):
    pass


def build_dataset(
    dataset: Dataset,
    dataset_config: DatasetConfig,
    featurizer_configs: List[AppmoleculefeaturizerLayerConfig] = None) -> DataModule:
    data_module = DataModule(dataset, dataset_config, featurizer_configs)
    return data_module
