from sqlalchemy.orm.session import Session

from app.features.dataset.crud import repo as dataset_repo
from app.features.dataset.model import Dataset


class TestDatasetCRUD:
    def test_get_by_name(self, db: Session, some_dataset: Dataset):
        dataset = dataset_repo.get_by_name(db, some_dataset.name)
        assert dataset.id == some_dataset.id
