from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql.schema import ForeignKey

from app.db.base_class import Base


class Model(Base):
    name = Column(String, primary_key=True, index=True)
    created_by_id = Column(Integer, ForeignKey("user.id", ondelete="SET NULL"))
    created_by = relationship(
        "User",
    )
