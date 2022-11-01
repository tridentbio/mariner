from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from mariner.db.base_class import Base


# class OAuthIntegration(Base):
#    __table__ = "oauth_integrations"
#    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
#    provider = Column(String, nullable=False)


class User(Base):
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=True)
    is_active = Column(Boolean(), default=True)
    is_superuser = Column(Boolean(), default=False)
    datasets = relationship("Dataset", back_populates="created_by")
    # oauth_integrations = relationship("OAuthIntegration")
