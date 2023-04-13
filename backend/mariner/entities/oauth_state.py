"""

"""
from sqlalchemy import Column, Integer, String

from mariner.db.base_class import Base


class OAuthState(Base):
    """
    Represents a server generated secret state used to validate
    requests to the oauth callback route.

    Requests with a state parameter not found in this table are
    from third-party requests, and should be ignored.
    """

    id = Column(Integer, primary_key=True)
    state = Column(String, index=True)
    provider = Column(String)
