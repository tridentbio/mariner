from app.features.model.components_query import (
    get_component_args_by_type,
    get_component_config_by_type,
)
from app.features.model.schema.layers_schema import (
    TorchlinearArgs,
    TorchlinearLayerConfig,
)


def test_get_component_config_by_type():
    component_config = get_component_config_by_type("torch.nn.Linear")
    assert component_config == TorchlinearLayerConfig


def test_get_component_args_by_type():
    component_config = get_component_args_by_type("torch.nn.Linear")
    assert component_config == TorchlinearArgs
