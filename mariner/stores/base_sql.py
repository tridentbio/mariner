"""
Base class and types for SQL persistence with sqlalchemy
"""
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy.orm import Session

from mariner.db.base_class import Base

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=Union[BaseModel, None])


class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Base class for all repositories with generic SQL queries implementation"""

    def __init__(self, model: Type[ModelType]):
        self.model = model

    def get(self, db: Session, id: Any) -> Optional[ModelType]:
        """Get a single entity with that id.

        Doesn't work for entities mapping to tables for which primary
        key is not an integer named "id".

        Args:
            db: Connection to the database.
            id: Id of the entity to get.

        Returns:
            The entity.
        """
        return db.query(self.model).filter(self.model.id == id).first()

    def get_multi(
        self, db: Session, *, skip: int = 0, limit: int = 100
    ) -> List[ModelType]:
        """Gets the items of a collection as a list.

        Args:
            db: Connection to the database.
            skip: Number of items to skip for pagination.
            limit: Number of items to be returned for pagination.

        Returns:
            The list with the results of the query.
        """
        return db.query(self.model).offset(skip).limit(limit).all()

    def create(self, db: Session, *, obj_in: CreateSchemaType) -> ModelType:
        """Persists an entity in the database.

        Args:
            db: Connection to the database.
            obj_in: The creation object

        Returns:
            The created object.
        """
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(**obj_in_data)  # type: ignore
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
        self,
        db: Session,
        *,
        db_obj: ModelType,
        obj_in: Union[UpdateSchemaType, Dict[str, Any]],
    ) -> ModelType:
        """Updates an entity.

        Args:
            db: Connection to the database.
            db_obj: Entity instance to be updated.
            obj_in: Update object

        Returns:
            The updated entity instance.
        """
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
        for (key, value) in update_data.items():
            dbfield = getattr(db_obj, key)
            if key != "id" and value != dbfield:
                setattr(db_obj, key, value)

        db.flush()
        db.commit()
        return db_obj

    def remove(self, db: Session, id: int) -> ModelType:
        """Removes an item of the collection by id.

        Args:
            db: Connection to the database.
            id: Id of the entity to be deleted.

        Returns:
            The deleted object.
        """
        obj = db.query(self.model).get(id)
        db.delete(obj)
        db.commit()
        return obj

    def get_paginated(
        self, db: Session, page: int, per_page: int
    ) -> tuple[List[ModelType], int]:
        """Similar to multi but also returns a count for the total number
        of items in the table.

        Args:
            db: Connection to database.
            page: Wanted page number.
            per_page: Number of items per page.

        Returns:
            A tuple with a list of results and a count.
        """
        sql_query = db.query(self.model)
        total = sql_query.count()
        sql_query = sql_query.limit(per_page)
        sql_query = sql_query.offset(page * per_page)
        result = sql_query.all()
        return result, total
