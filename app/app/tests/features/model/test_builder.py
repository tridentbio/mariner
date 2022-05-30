from app.features.model.builder import build_model_from_yaml
from app.features.model.featurizers import MoleculeFeaturizer

from app.tests.features.model.fixtures import create_example_model

featurizer = MoleculeFeaturizer()

# SMILES -> MoleculeFeaturizer -> GIN Conv x4 -> sum(node_features) -> linear -> Predicted LogD
def test_build_model():
    yaml_model = "app/tests/data/test_model.yaml"
    with open(yaml_model) as f:
        model = build_model_from_yaml(f.read())
        target_model = create_example_model()
        assert repr(model) == repr(target_model)

