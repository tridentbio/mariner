"""
Model related exceptions
"""

from typing import List


class ModelVersionNotFound(Exception):
    pass


class ModelNotFound(Exception):
    pass


class ModelNameAlreadyUsed(Exception):
    pass


class InvalidDataframe(Exception):
    def __init__(self, message: str, reasons: List[str]):
        super().__init__(message)
        self.reasons = reasons


class ModelVersionNotTrained(Exception):
    pass
