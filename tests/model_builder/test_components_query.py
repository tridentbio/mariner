from model_builder.components_query import (
    get_component_args_by_type,
    get_component_config_by_type,
)
from model_builder.layers_schema import TorchlinearLayerConfig, TorchlinearSummary


def test_get_component_config_by_type():
    component_config = get_component_config_by_type("torch.nn.Linear")
    assert component_config == TorchlinearLayerConfig


def test_get_component_args_by_type():
    component_config = get_component_args_by_type("torch.nn.Linear")
    assert component_config == TorchlinearSummary
