from sqlalchemy.orm.session import Session

from mariner.entities import Dataset
from mariner.schemas.dataset_schemas import DatasetUpdateRepo
from mariner.stores.dataset_sql import dataset_store
from tests.utils.utils import random_lower_string


class TestDatasetCRUD:
    def test_get_by_name(self, db: Session, some_dataset: Dataset):
        dataset = dataset_store.get_by_name(
            db, some_dataset.name, user_id=some_dataset.created_by_id
        )
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

        result = dataset_store.update(
            db, db_obj=some_dataset, obj_in=update_dataset_obj
        )
        for key, value in update_dataset_obj.dict().items():
            if key in update_data:
                assert getattr(result, key) == value
            else:
                assert getattr(result, key) == getattr(some_dataset, key)
