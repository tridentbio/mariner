from sqlalchemy.orm import Session

from app.crud.base import CRUDBase

from .model import Dataset
from .schema import DatasetCreateRepo, DatasetsQuery, DatasetUpdate

class CRUDDataset(CRUDBase[Dataset, DatasetCreateRepo, DatasetUpdate]):
    def get_many_paginated(self, db: Session, query: DatasetsQuery):
        sql_query = db.query(Dataset)

        ## sorting
        if query.sort_by_cols == 'ASC':
            sql_query = sql_query.order_by(Dataset.columns.asc())
        elif query.sort_by_cols == 'DESC':
            sql_query = sql_query.order_by(Dataset.columns.desc())

        if query.sort_by_rows == 'ASC':
            sql_query = sql_query.order_by(Dataset.columns.asc())
        elif query.sort_by_rows == 'DESC':
            sql_query = sql_query.order_by(Dataset.rows.desc())

        if query.sort_by_created_at == 'ASC':
            sql_query = sql_query.order_by(Dataset.created_at.asc())
        elif query.sort_by_created_at == 'DESC':
            sql_query = sql_query.order_by(Dataset.created_at.desc())

        ## filtering 
        if query.search_by_name:
            sql_query = sql_query.filter(Dataset.name.ilike(f'%{query.search_by_name}%'))
        if query.created_by_id:
            sql_query = sql_query.filter(Dataset.created_by_id == query.created_by_id)

        total = sql_query.count()
        print(query)
        sql_query = sql_query.limit(query.per_page).offset(
                query.page * query.per_page
        )
        result = sql_query.all()
        return result, total

    def create(self, db: Session, obj_in: DatasetCreateRepo):
        db_obj = Dataset(**obj_in.dict())
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

repo = CRUDDataset(Dataset)

