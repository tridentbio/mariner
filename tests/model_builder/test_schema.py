import pytest

from model_builder.schemas import ModelSchema
from tests.fixtures.model import model_config


@pytest.fixture(scope="module")
def schema_yaml_fixture():
    with open("tests/data/small_regressor_schema.yaml") as f:
        yield f.read()


def test_schema(schema_yaml_fixture: str):
    model_config = ModelSchema.from_yaml(schema_yaml_fixture)
    assert model_config
    assert model_config.layers[8].type == "model_builder.layers.Concat"


def test_schema_autofills_lossfn():
    regressor_schema = model_config(model_type="regressor")
    classifier_schema = model_config(model_type="classifier")
    assert regressor_schema.loss_fn == "torch.nn.MSELoss"
    assert classifier_schema.loss_fn == "torch.nn.CrossEntropyLoss"
