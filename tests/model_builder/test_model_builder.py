import pytest
import pytest_asyncio
from fastapi import UploadFile
from pytorch_lightning import Trainer
from sqlalchemy.orm.session import Session

from mariner import datasets as dataset_ctl
from mariner.entities import Dataset as DatasetEntity
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
from model_builder.model import CustomModel
from model_builder.schemas import ModelSchema
from tests.conftest import get_test_user
from tests.utils.utils import random_lower_string

# model configuration to be tested
model_config_yamls = [
    # "tests/data/small_regressor_schema.yaml",
    # "tests/data/categorical_features_model.yaml",
    "tests/data/dna_example.yml",
    # "tests/data/test_model_with_from_smiles.yaml",
]


@pytest_asyncio.fixture
async def zinc_extra_dataset(db: Session) -> Dataset:
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
        dataset = await dataset_ctl.create_dataset(
            db, get_test_user(db), dataset_create_obj
        )
        assert isinstance(
            dataset, DatasetEntity
        ), "creation of fixture dataset from zinc_extra.csv failed"
        dataset = Dataset.from_orm(dataset)
    assert dataset.id >= 1, "failed to create dataset"
    return dataset


@pytest.mark.parametrize("model_config_path", model_config_yamls)
@pytest.mark.asyncio
async def test_model_building(zinc_extra_dataset: Dataset, model_config_path: str):
    with open(model_config_path, "rb") as f:
        model_config = ModelSchema.from_yaml(f.read())
    model = CustomModel(model_config)
    data_module = DataModule(
        config=model_config,
        data=zinc_extra_dataset.get_dataframe(),
        split_target=zinc_extra_dataset.split_target,
        split_type=zinc_extra_dataset.split_type,
    )
    data_module.setup(stage="fit")
    trainer = Trainer(
        max_epochs=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, data_module.train_dataloader())
