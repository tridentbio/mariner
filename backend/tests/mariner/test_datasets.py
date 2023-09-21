from typing import Literal, Optional

import pytest
from fastapi import UploadFile
from sqlalchemy.orm import Session

from mariner import datasets as dataset_ctl
from mariner.entities import Dataset
from mariner.schemas.dataset_schemas import (
    ColumnsDescription,
    DatasetCreate,
    QuantityDataType,
    SmileDataType,
    Split,
    StringDataType,
)
from tests.fixtures.user import get_test_user
from tests.utils.utils import random_lower_string


def mock_dataset_create_request(
    split_type: Literal["scaffold", "random"] = "random",
    split_column: Optional[str] = None,
):
    colsdescription = [
        ColumnsDescription(
            pattern="CMPD_CHEM.*",
            data_type=StringDataType(domain_kind="string"),
            description="exp id",
        ),
        ColumnsDescription(
            pattern="exp",
            data_type=QuantityDataType(unit="mole", domain_kind="numeric"),
            description="exp id",
        ),
        ColumnsDescription(
            pattern="smiles",
            data_type=SmileDataType(domain_kind="smiles"),
            description="exp id",
        ),
    ]
    name = random_lower_string()
    create_obj = DatasetCreate.construct(
        name=name,
        columns_metadata=colsdescription,
        description=random_lower_string(),
        split_target=Split("80-10-10"),
        split_type=split_type,
        split_column=split_column,
    )
    assert create_obj.split_type == split_type
    return create_obj


@pytest.mark.parametrize(
    "create_obj",
    [
        mock_dataset_create_request(
            split_type="scaffold", split_column="smiles"
        ),
        mock_dataset_create_request(split_type="random"),
    ],
)
@pytest.mark.asyncio
@pytest.mark.integration
async def test_create_dataset(db: Session, create_obj: DatasetCreate):
    with open("tests/data/csv/Lipophilicity.csv", "rb") as f:
        user = get_test_user(db)
        create_obj.file = UploadFile(
            f,
            filename="file",
        )
        dataset = await dataset_ctl.create_dataset(db, user, create_obj)
        assert dataset
        assert dataset.name == create_obj.name
        db_obj = (
            db.query(Dataset).filter(Dataset.name == create_obj.name).first()
        )
        assert db_obj
        assert db_obj.id
