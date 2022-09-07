from sqlalchemy.orm.session import Session

from app.features.dataset.crud import repo as dataset_repo
from app.features.dataset.model import Dataset
from app.features.dataset.schema import DatasetUpdateRepo
from app.tests.utils.utils import random_lower_string


class TestDatasetCRUD:
    def test_get_by_name(self, db: Session, some_dataset: Dataset):
        dataset = dataset_repo.get_by_name(db, some_dataset.name)
        assert dataset.id == some_dataset.id

    def test_update(self, db: Session, some_dataset: Dataset):
        update_data = {
            "id": some_dataset.id,
            "name": random_lower_string(),
            "description": random_lower_string(),
            "rows": 10,
            "columns": 10,
            "bytes": 100,
            "columnsMetadata": [
                {"pattern": "abc", "dataType": "numeric", "description": "abc"}
            ],
        }
        update_dataset_obj = DatasetUpdateRepo(**update_data)

        result = dataset_repo.update(db, db_obj=some_dataset, obj_in=update_dataset_obj)
        for key, value in update_dataset_obj.dict().items():
            if key in update_data:
                assert getattr(result, key) == value
            else:
                assert getattr(result, key) == getattr(some_dataset, key)
