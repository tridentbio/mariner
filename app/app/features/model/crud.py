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


repo = CRUDModel(Model)
