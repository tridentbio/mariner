"""
Logging utility classes.
"""
import logging
from typing import List, Literal


class Filter(logging.Filter):
    """
    Filter out uvicorn logs.
    """

    def __init__(
        self,
        name: str,
        allowed_levels: None
        | List[Literal["INFO", "WARN", "ERROR", "DEBUG", "CRITICAL"]] = None,
        filter_messages: List[str] | None = None,
    ):
        super().__init__(name)
        self.name = name
        self.allowed_levels = allowed_levels
        self.filter_messages = filter_messages

    def filter(self, record):
        if not record.name.startswith(self.name):
            return True
        if (
            self.filter_messages
            and record.getMessage() in self.filter_messages
        ):
            return False
        if self.allowed_levels and record.levelname not in self.allowed_levels:
            return False
        return True
