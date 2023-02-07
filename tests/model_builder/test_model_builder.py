from typing import Callable, List

import pytest
import pytest_asyncio
import torch
from fastapi import UploadFile
from pytorch_lightning import Trainer
from sqlalchemy.orm.session import Session
from torch.nn.utils.rnn import pad_sequence

from mariner import datasets as dataset_ctl
from mariner.entities import Dataset as DatasetEntity
from mariner.schemas.dataset_schemas import (
    CategoricalDataType,
    ColumnsDescription,
    Dataset,
    DatasetCreate,
    DatasetCreateRepo,
    DNADataType,
    QuantityDataType,
    SmileDataType,
    Split,
)
from mariner.stores import dataset_sql
from model_builder.dataset import Collater, DataModule
from model_builder.model import CustomModel
from model_builder.schemas import ModelSchema
from tests.conftest import get_test_user
from tests.utils.utils import random_lower_string


def chem_dataset() -> tuple[str, List[ColumnsDescription]]:
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
    return zinc_extra_path, columns_metadata


def bio_dataset() -> tuple[str, List[ColumnsDescription]]:
    chimpazee_path = "tests/data/chimpanzee.csv"
    columns_metadata = [
        ColumnsDescription(
            pattern="sequence",
            data_type=DNADataType(),
            description="smiles column",
        ),
        ColumnsDescription(
            pattern=" class",
            data_type=CategoricalDataType(classes={str(i): i for i in range(0, 7)}),
            description="class",
        ),
    ]
    return chimpazee_path, columns_metadata


async def setup_dataset(
    db: Session, dataset_setup_fn: Callable[[], tuple[str, List[ColumnsDescription]]]
) -> Dataset:
    path, columns_descriptions = dataset_setup_fn()
    with open(path, "rb") as f:
        dataset_create_obj = DatasetCreate(
            name=random_lower_string(),
            description="Test description",
            split_type="random",
            split_target=Split("60-20-20"),
            file=UploadFile("file", f),
            columns_metadata=columns_descriptions,
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


# model configuration to be tested
model_configs_and_dataset_setup = [
    ("tests/data/small_regressor_schema.yaml", chem_dataset),
    ("tests/data/categorical_features_model.yaml", chem_dataset),
    ("tests/data/dna_example.yml", bio_dataset),
]


@pytest.mark.parametrize(
    "model_config_path,dataset_setup", model_configs_and_dataset_setup
)
@pytest.mark.asyncio
async def test_model_building(
    db: Session, model_config_path: str, dataset_setup: Callable
):
    dataset = await setup_dataset(db, dataset_setup)
    with open(model_config_path, "rb") as f:
        model_config = ModelSchema.from_yaml(f.read())
    model = CustomModel(model_config)
    collate_fn = Collater()
    data_module = DataModule(
        config=model_config,
        data=dataset.get_dataframe(),
        split_target=dataset.split_target,
        split_type=dataset.split_type,
        collate_fn=collate_fn,
        batch_size=4,
    )
    data_module.setup(stage="fit")
    trainer = Trainer(
        max_epochs=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, data_module.train_dataloader())
