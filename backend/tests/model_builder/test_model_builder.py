from typing import Callable, List

import pytest
import torch
from fastapi import UploadFile
from lightning.pytorch import Trainer
from sqlalchemy.orm.session import Session

from mariner import datasets as dataset_ctl
from mariner.entities import Dataset as DatasetEntity
from mariner.schemas.dataset_schemas import (
    CategoricalDataType,
    ColumnsDescription,
    Dataset,
    DatasetCreate,
    DNADataType,
    QuantityDataType,
    SmileDataType,
    Split,
)
from fleet.model_builder.dataset import Collater, DataModule
from fleet.model_builder.model import CustomModel
from fleet.model_builder.optimizers import AdamOptimizer
from fleet.model_builder.schemas import ModelSchema
from tests.conftest import get_test_user
from tests.utils.utils import random_lower_string


def chem_dataset() -> tuple[str, List[ColumnsDescription]]:
    zinc_extra_path = "tests/data/csv/zinc_extra.csv"
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


def bio_classification_dataset() -> tuple[str, List[ColumnsDescription]]:
    chimpazee_path = "tests/data/csv/chimpanzee.csv"
    columns_metadata = [
        ColumnsDescription(
            pattern="sequence",
            data_type=DNADataType(),
            description="dna sequence",
        ),
        ColumnsDescription(
            pattern=" class",
            data_type=CategoricalDataType(classes={str(i): i for i in range(0, 7)}),
            description="class",
        ),
    ]
    return chimpazee_path, columns_metadata


def bio_regression_dataset() -> tuple[str, List[ColumnsDescription]]:
    # aaMutations,uniqueBarcodes,medianBrightness,std
    # DNA,int,float,float
    csvpath = "tests/data/csv/sarkisyan_full_seq_data.csv"
    columns = [
        ColumnsDescription(
            pattern="aaMutations",
            data_type=DNADataType(),
            description="dna sequence",
        ),
        ColumnsDescription(
            pattern="medianBrightness",
            data_type=QuantityDataType(unit="mole"),
            description="median brightness",
        ),
    ]
    return csvpath, columns


def iris_dataset() -> tuple[str, List[ColumnsDescription]]:
    irirs_path = "tests/data/csv/iris.csv"
    columns_metadata = [
        ColumnsDescription(
            pattern="petal_length",
            data_type=QuantityDataType(unit="mole"),
            description="petal length",
        ),
        ColumnsDescription(
            pattern="sepal_length",
            data_type=QuantityDataType(unit="mole"),
            description="sepal length",
        ),
        ColumnsDescription(
            pattern="sepal_width",
            data_type=QuantityDataType(unit="mole"),
            description="sepal width",
        ),
        ColumnsDescription(
            pattern="species",
            data_type=CategoricalDataType(
                classes={"setosa": 0, "versicolor": 1, "virginica": 2}
            ),
            description="species",
        ),
        ColumnsDescription(
            pattern="large_petal_length",
            data_type=CategoricalDataType(classes={"yes": 1, "no": 0}),
            description="yes if petal length is larger than 5.10 otherwise no",
        ),
        ColumnsDescription(
            pattern="large_petal_width",
            data_type=CategoricalDataType(classes={"yes": 1, "no": 0}),
            description="yes if petal width is larger than 1.80 otherwise no",
        ),
    ]
    return irirs_path, columns_metadata


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
    ("tests/data/yaml/small_regressor_schema.yaml", chem_dataset),
    ("tests/data/yaml/categorical_features_model.yaml", chem_dataset),
    ("tests/data/yaml/dna_example.yml", bio_regression_dataset),
    (
        "tests/data/yaml/binary_classification_model.yaml",
        iris_dataset,
        8,
    ),
    (
        "tests/data/yaml/multiclass_classification_model.yaml",
        iris_dataset,
        8,
    ),
    (
        "tests/data/yaml/multitarget_classification_model.yaml",
        iris_dataset,
        8,
    ),
]

# setting default batch_size=4 for model configs that don't have them
model_configs_and_dataset_setup = list(
    map(lambda x: x if len(x) == 3 else (*x, 4), model_configs_and_dataset_setup)
)


@pytest.mark.parametrize(
    "model_config_path,dataset_setup,batch_size", model_configs_and_dataset_setup
)
@pytest.mark.asyncio
@pytest.mark.integration
async def test_model_building(
    db: Session, model_config_path: str, dataset_setup: Callable, batch_size: int
):
    dataset = await setup_dataset(db, dataset_setup)
    with open(model_config_path, "rb") as f:
        model_config = ModelSchema.from_yaml(f.read())
    model = CustomModel(model_config)
    model.set_optimizer(optimizer=AdamOptimizer())
    data_module = DataModule(
        config=model_config,
        data=dataset.get_dataframe(),
        split_target=dataset.split_target,
        split_type=dataset.split_type,
        collate_fn=Collater(),
        batch_size=batch_size,
    )
    data_module.setup(stage="fit")
    trainer = Trainer(
        max_epochs=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, data_module.train_dataloader())

    batch_example = next(iter(data_module.val_dataloader()))
    foward = model(batch_example)
    predict = model.predict_step(batch_example, 0)

    for target_column in model_config.dataset.target_columns:
        if target_column.column_type == "binary":
            assert torch.allclose(
                torch.sigmoid(foward[target_column.name].squeeze()),
                predict[target_column.name],
            ), (
                "predict_step is not equal to sigmoid of forward for "
                f"binary classifier column {target_column.name}"
            )
        elif target_column.column_type == "multiclass":
            assert torch.equal(
                torch.nn.functional.softmax(
                    foward[target_column.name].squeeze(), dim=-1
                ),
                predict[target_column.name],
            ), (
                "predict_step is not equal to softmax of forward for "
                f"multi class classifier column {target_column.name}"
            )
