"""
Base class for entities mapping to table name
"""
from typing import Any

from sqlalchemy.ext.declarative import as_declarative, declared_attr


@as_declarative()
class Base:
    """Base class for mapping entities to SQL tables"""

    id: Any
    __name__: str

    # Generate __tablename__ automatically
    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower().removesuffix("entity").rstrip("_")
