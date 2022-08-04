from subprocess import CalledProcessError, check_output

import pytest

from app.features.model import generate
from app.features.model.schema.layers_schema import (
    TorchlinearArgs,
    TorchlinearLayerConfig,
)


def test_generate_bundle():
    python_code = generate.generate_bundle()
    assert isinstance(python_code, str)
    try:
        check_output(["python"], input=python_code, text=True)
    except CalledProcessError:
        pytest.fail("Failed to exectute generated python bundle`")


def test_get_component_config_by_type():
    component_config = generate.get_component_config_by_type("torch.nn.Linear")
    assert component_config == TorchlinearLayerConfig


def test_get_component_args_by_type():
    component_config = generate.get_component_args_by_type("torch.nn.Linear")
    assert component_config == TorchlinearArgs
