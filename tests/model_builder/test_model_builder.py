from typing import Generator

import pytest
from fastapi import UploadFile
from pytorch_lightning import Trainer
from sqlalchemy.orm.session import Session

from mariner import datasets as dataset_ctl
from mariner.schemas.dataset_schemas import (
    CategoricalDataType,
    ColumnsDescription,
    Dataset,
    DatasetCreate,
    QuantityDataType,
    SmileDataType,
    Split,
)
from model_builder.dataset import DataModule
from model_builder.model import CustomModel, CustomModelV2
from model_builder.schemas import ModelSchema
from tests.conftest import get_test_user
from tests.utils.utils import random_lower_string

# model configuration to be tested
mlflow_yamls = [
    "tests/data/test_model_hard.yaml",
    # "app/tests/data/categorical_features_model.yaml",
]


@pytest.fixture()
def zinc_extra_dataset(db: Session) -> Generator[Dataset, None, None]:
    zinc_extra_path = "tests/data/zinc_extra.csv"
    columns_metadata = [
        ColumnsDescription(
            pattern="smiles",
            data_type=SmileDataType(domain_kind="smiles"),
            description="smiles column",
        ),
        ColumnsDescription(
            pattern="mwt",
            data_type=QuantityDataType(domain_kind="numeric", unit="mole"),
            description="Molecular Weigth",
        ),
        ColumnsDescription(
            pattern="tpsa",
            data_type=QuantityDataType(domain_kind="numeric", unit="mole"),
            description="T Polar surface",
        ),
        ColumnsDescription(
            pattern="mwt_group",
            data_type=CategoricalDataType(
                domain_kind="categorical", classes={"yes": 0, "no": 1}
            ),
            description="yes if mwt is larger than 300 otherwise no",
        ),
    ]
    with open(zinc_extra_path, "rb") as f:
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


@pytest.mark.parametrize("model_config_path", mlflow_yamls)
def test_model_building(zinc_extra_dataset: Dataset, model_config_path: str):
    with open(model_config_path, "rb") as f:
        model_config = ModelSchema.from_yaml(f.read())
    model = CustomModelV2(model_config)
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
