import yaml
from sqlalchemy.orm.session import Session

from app.features.dataset.crud import repo as datasetsrepo
from app.features.model.builder import build_dataset, build_model_from_yaml
from app.features.model.schema import ModelConfig, MoleculeFeaturizerConfig


def test_dataset_creation(db: Session):
    yaml_model = "app/tests/data/test_model.yaml"
    with open(yaml_model) as f:
        config_dict = yaml.safe_load(f.read())
        config = ModelConfig.parse_obj(config_dict)
        dataset = build_dataset(config, datasetsrepo, db)
        item = dataset[0]
        print(item)
        assert item is not None


def test_model_schema():
    yaml_model = "app/tests/data/test_model.yaml"
    with open(yaml_model) as f:
        model = ModelConfig.from_yaml(f.read())
        assert isinstance(model, ModelConfig)
        assert isinstance(model.featurizer, MoleculeFeaturizerConfig)
        feat = model.featurizer.create()
        assert feat("CCCC") is not None


def test_model_trains():
    yaml_model = "app/tests/data/test_model.yaml"
    with open(yaml_model) as f:
        build_model_from_yaml(f.read())
        # train(model, epochs=2, learning_rate=0.1)
