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


def test_schema_1():
    schema = """
name: Simple Classifier
dataset:
  name: zinc dataset
  targetColumn: 
    name: mwt_group
    dataType:
      domainKind: categorical
      classes:
        mwt_small: 0
        mwt_big: 1
  featureColumns:
    - name: mwt
      dataType:
        domainKind: numeric

featurizers:
  - name: "MWT Cat Featurizer"
    type: model_builder.featurizers.IntegerFeaturizer
    forwardArgs:
      input: $mwt_group
      
layers:
  - name: "Linear1"
    type: torch.nn.Linear
    constructorArgs:
      in_features: 1
      out_features: 2
    forwardArgs:
      input: $mwt
  - name: "Sigmoid1"
    type: torch.nn.Sigmoid
    forwardArgs:
      input: $Linear1
    """
    schema = ModelSchema.from_yaml(schema)
    assert isinstance(schema, ModelSchema)
