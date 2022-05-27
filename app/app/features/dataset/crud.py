from typing import Any, Dict, Union

from sqlalchemy.orm import Session

from app.crud.base import CRUDBase

from .model import ColumnDescription, Dataset
from .schema import (
    DatasetCreateRepo,
    DatasetsQuery,
    DatasetUpdate,
    DatasetUpdateRepo,
)


class CRUDDataset(CRUDBase[Dataset, DatasetCreateRepo, DatasetUpdate]):
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
        ds_data = {
            k: obj_in_dict[k] for k in obj_in_dict if k != "columns_descriptions"
        }
        db_obj = Dataset(**ds_data)

        if obj_in.columns_descriptions:
            db_obj.columns_descriptions = [
                ColumnDescription(
                    pattern=cd_in["pattern"], description=cd_in["description"]
                )
                for cd_in in obj_in_dict["columns_descriptions"]
            ]

        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
        self,
        db: Session,
        db_obj: Dataset,
        obj_in: Union[DatasetUpdateRepo, Dict[str, Any]],
    ):
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
        super().update(db, db_obj=db_obj, obj_in=update_data)
        return db_obj


repo = CRUDDataset(Dataset)
