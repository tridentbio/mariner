"""
User entity
"""

from sqlalchemy import Boolean, Column, Integer, String
from sqlalchemy.orm import relationship

from mariner.db.base_class import Base


class User(Base):
    """Entity mapping to a user of the mariner application."""

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=True)
    is_active = Column(Boolean(), default=True)
    is_superuser = Column(Boolean(), default=False)
    img_url = Column(String, nullable=True)
    datasets = relationship("Dataset", back_populates="created_by")
