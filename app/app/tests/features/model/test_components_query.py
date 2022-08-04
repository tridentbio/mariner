from app.features.model.generate import generate
from app.features.model.schema.layers_schema import (
    TorchlinearArgs,
    TorchlinearLayerConfig,
)


def test_get_component_config_by_type():
    component_config = generate.get_component_config_by_type("torch.nn.Linear")
    assert component_config == TorchlinearLayerConfig


def test_get_component_args_by_type():
    component_config = generate.get_component_args_by_type("torch.nn.Linear")
    assert component_config == TorchlinearArgs
