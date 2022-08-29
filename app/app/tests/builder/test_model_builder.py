from typing import Generator
from fastapi import UploadFile
import pytest
from pytorch_lightning import Trainer
from sqlalchemy.orm.session import Session
from app.builder.dataset import DataModule
from app.builder.model import CustomModel

from app.features.dataset.schema import CategoricalDataType, Dataset, DatasetCreate, ColumnsDescription, NumericalDataType, SmileDataType, Split
from app.features.dataset import controller as dataset_ctl
from app.features.model.schema.configs import ModelConfig
from app.tests.conftest import get_test_user
from app.tests.utils.utils import random_lower_string

# model configuration to be tested
mlflow_yamls = [
    "app/tests/data/test_model_hard.yaml",
    "app/tests/data/categorical_features_model.yaml",
]

@pytest.fixture()
def zinc_extra_dataset(db: Session) -> Generator[Dataset, None, None]:
    zinc_extra_path = 'app/tests/data/zinc_extra.csv'
    columns_metadata = [
        ColumnsDescription(
            pattern="smiles",
            data_type=SmileDataType(),
            description="smiles column",
        ),
        ColumnsDescription(
            pattern="mwt",
            data_type=NumericalDataType(),
            description="Molecular Weigth",
        ),
        ColumnsDescription(
            pattern="tpsa",
            data_type=NumericalDataType(),
            description="T Polar surface",
        ),
        ColumnsDescription(
            pattern="mwt_group",
            data_type=CategoricalDataType(classes={
                'yes': 0,
                'no': 1
            }),
            description="yes if mwt is larger than 300 otherwise no",
        ),
    ]
    with open(zinc_extra_path, 'rb') as f:
        dataset_create_obj = DatasetCreate(
            name=random_lower_string(),
            description="Test description",
            split_type="random",
            split_target=Split("60-20-20"),
            file=UploadFile("file", f),
            columns_metadata=columns_metadata,
        )
        dataset = dataset_ctl.create_dataset(db, get_test_user(db), dataset_create_obj)
        assert dataset, "creation of fixture dataset from zinc_extra.csv failed"
        dataset = Dataset.from_orm(dataset)
        yield dataset

@pytest.mark.parametrize('model_config_path', mlflow_yamls)
def test_model_building(zinc_extra_dataset: Dataset, model_config_path: str):
    with open(model_config_path, 'rb') as f:
        model_config = ModelConfig.from_yaml(f.read())
    model = CustomModel(model_config)
    featurizers_config = model_config.featurizers
    data_module = DataModule(
        featurizers_config=featurizers_config,
        data=zinc_extra_dataset.get_dataframe(),
        dataset_config=model_config.dataset,
        split_target=zinc_extra_dataset.split_target,
        split_type=zinc_extra_dataset.split_type,
    )
    data_module.setup(stage="fit")
    trainer = Trainer(
        max_epochs=2,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, data_module.train_dataloader())

