"""
Database connection functions
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mariner.core.config import get_app_settings

engine = create_engine(
    str(get_app_settings("services").postgres_uri), pool_pre_ping=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
