"""
Model data layer defining ways to read and write to the models and model
versions collection
"""

from typing import List, Optional

from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm.session import Session
from sqlalchemy.sql import and_

from mariner.entities.model import Model, ModelFeaturesAndTarget, ModelVersion
from mariner.schemas.model_schemas import (
    ModelCreateRepo,
    ModelVersionCreateRepo,
    ModelVersionUpdateRepo,
)
from mariner.stores.base_sql import CRUDBase


class CRUDModel(CRUDBase[Model, ModelCreateRepo, None]):
    """Data layer operators on the models collection"""

    def get_by_name_from_user(self, db: Session, name: str, user_id: int) -> Model:
        """Gets a model by name and created_by_id.

        Args:
            db: Connection to the database.
            user_id: Id of the user to filter models by creator.
            name: Name of the model.

        Returns:
            The model instance or None if no result is found.
        """
        return (
            db.query(Model)
            .filter(and_(Model.name == name, Model.created_by_id == user_id))
            .first()
        )

    def get_paginated(
        self,
        db: Session,
        page: int,
        per_page: int,
        created_by_id: Optional[int] = None,
        q: Optional[str] = None,
    ) -> tuple[List[Model], int]:
        """[TODO:summary]

        Gets a paginated view of the models.

        Args:
            db: Connection to the database.
            page: number of the page to get.
            per_page: number of items per page.
            created_by_id: id of user that created the model.
            q: string used to filter models by name.

        Returns:
            The paginated result with the found models.
        """
        sql_query = db.query(Model)
        if created_by_id:
            sql_query = sql_query.filter(Model.created_by_id == created_by_id)
        if q:
            sql_query = sql_query.filter(Model.name.startswith(q))
        total = sql_query.count()
        sql_query = sql_query.limit(per_page)
        sql_query = sql_query.offset(per_page * page)
        result = sql_query.all()
        return result, total

    def create_model_version(
        self, db: Session, version_create: ModelVersionCreateRepo
    ) -> ModelVersion:
        """Creates a model version.

        Args:
            db: Connection to the database.
            version_create: Model version creation object.

        Returns:
            The created model version.
        """
        obj_in_data = jsonable_encoder(version_create)
        db_obj = ModelVersion(**obj_in_data)  # type: ignore
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def create(self, db: Session, obj_in: ModelCreateRepo):
        """Creates a model.

        Args:
            db: Connection to the database.
            obj_in: model creation object.
        """
        relations = ["columns"]
        obj_in_dict = obj_in.dict()
        model_data = {k: obj_in_dict[k] for k in obj_in_dict if k not in relations}
        db_obj = Model(**model_data)

        if obj_in.columns:
            db_obj.columns = [
                ModelFeaturesAndTarget(
                    column_name=col.column_name, column_type=col.column_type
                )
                for col in obj_in.columns
            ]

        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)

        return db_obj

    def delete_by_id(self, db: Session, model_id: int) -> None:
        """Deletes a single model by id.

        Args:
            db: Connection to the database.
            model_id: Id of the model to be deleted.
        """
        db.query(Model).filter(Model.id == model_id).delete()
        db.commit()

    def get_model_version(self, db: Session, id: int) -> Optional[ModelVersion]:
        """Get's a single model version by id.

        Args:
            db: Connection to the database.
            id: Integer id to be used in query.

        Returns:
            Model version instance with given id or None if none is found.
        """
        return db.query(ModelVersion).filter(ModelVersion.id == id).first()

    def update_model_version(
        self, db: Session, version_id: int, obj_in: ModelVersionUpdateRepo
    ):
        """Updates model version

        Args:
            db: Connection to the database
            version_id: id of the version to be updated.
            obj_in: version database entity instance.
        """
        version = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
        for key, value in obj_in.dict().items():
            setattr(version, key, value)
        db.add(version)
        db.flush()


model_store = CRUDModel(Model)
