from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.features.dataset import controller as dataset_ctl
from app.features.dataset.model import Dataset
from app.features.dataset.schema import (
    ColumnMetadata,
    DatasetCreate,
    DataType,
    Split,
)
from app.tests.conftest import get_test_user
from app.tests.utils.utils import random_lower_string


def test_create_dataset(db: Session):
    with open("app/tests/data/Lipophilicity.csv", "rb") as f:
        colsdescription = [
            ColumnMetadata(
                pattern="CMPD_CHEM.*",
                data_type=DataType("string"),
                description="exp id",
            ),
            ColumnMetadata(
                pattern="exp", data_type=DataType("numerical"), description="exp id"
            ),
            ColumnMetadata(
                pattern="smiles", data_type=DataType("smiles"), description="exp id"
            ),
        ]
        name = random_lower_string()
        create_obj = DatasetCreate(
            file=UploadFile("file", f),
            name=name,
            columns_metadata=colsdescription,
            description=random_lower_string(),
            split_target=Split("80-10-10"),
            split_type="random",
        )
        user = get_test_user(db)
        dataset = dataset_ctl.create_dataset(db, user, create_obj)
        assert dataset.name == name
        db_obj = db.query(Dataset).filter(Dataset.name == name).first()
        assert db_obj
        assert db_obj.id
