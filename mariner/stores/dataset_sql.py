from sqlalchemy.orm import Session
from mariner.entities.dataset import ColumnsMetadata, Dataset

from mariner.stores.base_sql import CRUDBase
from mariner.schemas.dataset_schemas import (
    DatasetCreateRepo,
    DatasetsQuery,
    DatasetUpdateRepo,
)


class CRUDDataset(CRUDBase[Dataset, DatasetCreateRepo, DatasetUpdateRepo]):
    def get_many_paginated(self, db: Session, query: DatasetsQuery):
        sql_query = db.query(Dataset)

        # sorting
        if query.sort_by_cols == "asc":
            sql_query = sql_query.order_by(Dataset.columns.asc())
        elif query.sort_by_cols == "desc":
            sql_query = sql_query.order_by(Dataset.columns.desc())

        if query.sort_by_rows == "asc":
            sql_query = sql_query.order_by(Dataset.columns.asc())
        elif query.sort_by_rows == "desc":
            sql_query = sql_query.order_by(Dataset.rows.desc())

        if query.sort_by_created_at == "asc":
            sql_query = sql_query.order_by(Dataset.created_at.asc())
        elif query.sort_by_created_at == "desc":
            sql_query = sql_query.order_by(Dataset.created_at.desc())

        # filtering
        if query.search_by_name:
            sql_query = sql_query.filter(
                Dataset.name.ilike(f"%{query.search_by_name}%")
            )
        if query.created_by_id:
            sql_query = sql_query.filter(Dataset.created_by_id == query.created_by_id)

        total = sql_query.count()
        sql_query = sql_query.limit(query.per_page).offset(query.page * query.per_page)
        result = sql_query.all()
        return result, total

    def create(self, db: Session, obj_in: DatasetCreateRepo):
        obj_in_dict = obj_in.dict()
        relations_key = ["columns_metadata"]
        ds_data = {k: obj_in_dict[k] for k in obj_in_dict if k not in relations_key}
        db_obj = Dataset(**ds_data)
        if obj_in.columns_metadata:
            db_obj.columns_metadata = [
                ColumnsMetadata(**cd_me) for cd_me in obj_in_dict["columns_metadata"]
            ]

        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
        self,
        db: Session,
        db_obj: Dataset,
        obj_in: DatasetUpdateRepo,
    ):
        if obj_in.columns_metadata:
            db.query(ColumnsMetadata).filter(
                ColumnsMetadata.dataset_id == db_obj.id
            ).delete()
            db_obj.columns_metadata = [
                ColumnsMetadata(**cd_me.dict()) for cd_me in obj_in.columns_metadata
            ]
            db.add(db_obj)
            del obj_in.columns_metadata
        super().update(db, db_obj=db_obj, obj_in=obj_in)
        return db_obj

    def get_by_name(self, db: Session, name: str) -> Dataset:
        return db.query(Dataset).filter(Dataset.name == name).first()


dataset_store = CRUDDataset(Dataset)
