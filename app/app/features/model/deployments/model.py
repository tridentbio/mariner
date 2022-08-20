from sqlalchemy.orm import relationship
from sqlalchemy.sql.schema import Column, ForeignKey
from sqlalchemy.sql.sqltypes import Integer, String

from app.db.base_class import Base


class Deployment(Base):
    name = Column(String, primary_key=True)
    model_version_id = Column(
        Integer, ForeignKey("modelversion.id", ondelete="CASCADE")
    )
    created_by_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"))
    created_by = relationship("User")
    model = relationship("ModelVersion")
