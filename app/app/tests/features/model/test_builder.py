import yaml
from sqlalchemy.orm.session import Session
from torch import nn
from torch_geometric.data import DataLoader

from app.features.dataset.crud import repo as datasetsrepo
from app.features.model.builder import CustomGraphModel, build_dataset
from app.features.model.schema.configs import (
    ModelConfig,
    MoleculeFeaturizerConfig,
)


def test_dataset(db: Session):
    yaml_model = "app/tests/data/test_model.yaml"
    with open(yaml_model) as f:
        config_dict = yaml.safe_load(f.read())
        config = ModelConfig.parse_obj(config_dict)
        dataset = build_dataset(config, datasetsrepo, db)
        item = dataset[0]
        assert item is not None
        # TODO: assert over item data


def test_model_schema():
    yaml_model = "app/tests/data/test_model.yaml"
    with open(yaml_model) as f:
        model = ModelConfig.from_yaml(f.read())
        assert isinstance(model, ModelConfig)
        assert isinstance(model.featurizer, MoleculeFeaturizerConfig)
        feat = model.featurizer.create()
        assert feat("CCCC") is not None


def test_model_build(db: Session):
    yaml_model = "app/tests/data/test_model.yaml"
    with open(yaml_model) as f:
        model_config = ModelConfig.from_yaml(f.read())
        ds = build_dataset(model_config, datasetsrepo, db)
        model = CustomGraphModel(model_config)
        assert isinstance(model, nn.Module)
        loader = DataLoader(ds)
        batch = next(iter(loader))
        out = model(batch)
        print(out)
        assert out is not None
        # assert out.size() == y.size()
        # train(model, epochs=2, learning_rate=0.1)
