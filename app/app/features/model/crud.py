from typing import List, Optional
from sqlalchemy.orm.session import Session

from app.crud.base import CRUDBase
from app.features.model.model import Model
from app.features.model.schema.model import ModelCreateRepo, ModelUpdateRepo


class CRUDModel(CRUDBase[Model, ModelCreateRepo, ModelUpdateRepo]):
    def get_by_name(self, db: Session, name: str) -> Model:
        return db.query(Model).filter(Model.name == name).first()
    def remove_by_name(self, db: Session, name: str):
        obj = db.query(Model).filter(Model.name == name).first()
        if obj:
            db.delete(obj)
            db.commit()
        return obj
    def get_paginated(self, db: Session, page: int, per_page: int, created_by_id: Optional[int] = None) -> tuple[List[Model], int]:
        sql_query = db.query(Model)
        if created_by_id:
            sql_query = sql_query.filter(Model.created_by_id == created_by_id)
        total = sql_query.count()
        sql_query = sql_query.limit(per_page)
        sql_query = sql_query.offset(per_page * page)
        result = sql_query.all()
        return result, total



repo = CRUDModel(Model)
