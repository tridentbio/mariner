from os import name
from typing import List, Optional

from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm.session import Session

from app.crud.base import CRUDBase
from app.features.dataset.schema import ColumnsMeta
from app.features.model.model import Model, ModelFeaturesAndTarget, ModelVersion
from app.features.model.schema.model import (
    ModelCreateRepo,
    ModelUpdateRepo,
    ModelVersionCreateRepo,
)


class CRUDModel(CRUDBase[Model, ModelCreateRepo, ModelUpdateRepo]):
    def get_by_name(self, db: Session, name: str) -> Model:
        return db.query(Model).filter(Model.name == name).first()

    def remove_by_name(self, db: Session, name: str):
        obj = db.query(Model).filter(Model.name == name).first()
        if obj:
            db.delete(obj)
            db.commit()
        return obj

    def get_by_name_from_user(self, db: Session, user_id: int, name: str):
        model = (
            db.query(Model)
            .filter(Model.created_by_id == user_id)
            .filter(Model.name == name)
            .first()
        )
        return model

    def get_paginated(
        self, db: Session, page: int, per_page: int, created_by_id: Optional[int] = None
    ) -> tuple[List[Model], int]:
        sql_query = db.query(Model)
        if created_by_id:
            sql_query = sql_query.filter(Model.created_by_id == created_by_id)
        total = sql_query.count()
        sql_query = sql_query.limit(per_page)
        sql_query = sql_query.offset(per_page * page)
        result = sql_query.all()
        return result, total

    def create_model_version(
        self, db: Session, version_create: ModelVersionCreateRepo
    ) -> ModelVersion:
        obj_in_data = jsonable_encoder(version_create)
        db_obj = ModelVersion(**obj_in_data)  # type: ignore
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def create(self, db: Session, obj_in: ModelCreateRepo):
        relations = ["columns"]
        obj_in_dict = obj_in.dict()
        model_data = {k: obj_in_dict[k] for k in obj_in_dict if k not in relations}
        db_obj = Model(**model_data)

        if obj_in.columns:
            db_obj.columns = [
                ModelFeaturesAndTarget(**col.dict()) for col in obj_in.columns
            ]

        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)

        return db_obj


repo = CRUDModel(Model)
