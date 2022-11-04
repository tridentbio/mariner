import pytest
import yaml

from model_builder.layers_schema import ModelbuilderconcatForwardArgsReferences
from model_builder.schemas import ModelSchema


def test_concat_forward_args():
    y = """
name: Combiner
type: model_builder.layers.Concat
forwardArgs: 
  x1: "${AddPool}"
  x2: "${Linear1}"
"""
    json = yaml.unsafe_load(y)
    assert "forwardArgs" in json
    assert "x2" in json["forwardArgs"]
    assert ModelbuilderconcatForwardArgsReferences.parse_obj(
        json["forwardArgs"]
    ) == ModelbuilderconcatForwardArgsReferences(x1="${AddPool}", x2="${Linear1}")


@pytest.fixture(scope="module")
def schema_yaml_fixture():
    with open("tests/data/test_model_hard.yaml") as f:
        yield f.read()


def test_schema(schema_yaml_fixture: str):
    model_config = ModelSchema.from_yaml(schema_yaml_fixture)
    assert model_config
    assert model_config.layers[8].type == "model_builder.layers.Concat"
