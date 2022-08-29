from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.features.dataset import controller as dataset_ctl
from app.features.dataset.model import Dataset
from app.features.dataset.schema import (
    ColumnsDescription,
    DatasetCreate,
    NumericalDataType,
    SmileDataType,
    Split,
    StringDataType,
)
from app.tests.conftest import get_test_user
from app.tests.utils.utils import random_lower_string


def test_create_dataset(db: Session):
    with open("app/tests/data/Lipophilicity.csv", "rb") as f:
        colsdescription = [
            ColumnsDescription(
                pattern="CMPD_CHEM.*",
                data_type=StringDataType(domain_kind="string"),
                description="exp id",
            ),
            ColumnsDescription(
                pattern="exp", data_type=NumericalDataType(domain_kind="numerical"), description="exp id"
            ),
            ColumnsDescription(
                pattern="smiles", data_type=SmileDataType(domain_kind="smiles"), description="exp id"
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
        assert dataset
        assert dataset.name == name
        db_obj = db.query(Dataset).filter(Dataset.name == name).first()
        assert db_obj
        assert db_obj.id
