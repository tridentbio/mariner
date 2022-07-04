import pytest
import yaml
from sqlalchemy.orm.session import Session
from torch import nn
from torch_geometric.data import DataLoader
from pytorch_lightning import Trainer

from app.features.dataset.crud import repo as datasetsrepo
from app.features.model.builder import CustomModel, build_dataset
from app.features.model.schema.configs import ModelConfig


def test_dataset(db: Session):
    yaml_model = "app/tests/data/test_model_hard.yaml"
    with open(yaml_model) as f:
        config_dict = yaml.safe_load(f.read())
        config = ModelConfig.parse_obj(config_dict)
        dataset = build_dataset(config, datasetsrepo, db)
        item = dataset[0]
        assert item is not None
        # TODO: assert over item data


def test_model_schema():
    yaml_model = "app/tests/data/test_model_hard.yaml"
    with open(yaml_model) as f:
        model = ModelConfig.from_yaml(f.read())
        assert isinstance(model, ModelConfig)


def test_model_build(db: Session):
    yaml_model = "app/tests/data/test_model_hard.yaml"
    with open(yaml_model) as f:
        model_config = ModelConfig.from_yaml(f.read())
        ds = build_dataset(model_config, datasetsrepo, db)
        model = CustomModel(model_config)
        assert isinstance(model, nn.Module)
        loader = DataLoader(ds)
        x, y = next(iter(loader))
        out = model(x)
        assert out is not None
        trainer = Trainer(max_epochs=2)
        trainer.fit(model, loader)
        ## Trains without error
        # TODO: check against another implementation of same model
