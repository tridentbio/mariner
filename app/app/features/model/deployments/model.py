from sqlalchemy.orm import relationship
from sqlalchemy.sql.schema import Column, ForeignKey
from sqlalchemy.sql.sqltypes import Integer, String

from app.db.base_class import Base


class Deployment(Base):
    name = Column(String, primary_key=True)
    model_name = Column(String, ForeignKey("model.name", ondelete="SET NULL"))
    model_version = Column(String)
    created_by_id = Column(Integer, ForeignKey("user.id", ondelete="SET NULL"))
    created_by = relationship("User")
    model = relationship("Model")
