"""
Model related exceptions
"""

from typing import List


class ModelVersionNotFound(Exception):
    """Raised when the application can't proceed because of
    the absence of a model version."""


class ModelNotFound(Exception):
    """Raised when the application can't proceed because of
    the absence of a model."""


class ModelNameAlreadyUsed(Exception):
    """Raised when user repeats model name.

    It's not allowed to have 2 models because it makes the name
    attribute harder to be used for specifying a model.
    """


class InvalidDataframe(Exception):
    """Raised when a dataframe given for prediction does not conform
    to training dataset."""

    def __init__(self, message: str, reasons: List[str]):
        super().__init__(message)
        self.reasons = reasons


class ModelVersionNotTrained(Exception):
    """Raised when the application can't proceed because the
    user is trying to use a model not yet trained.
    """


class ModelVersionNotUpdatable(Exception):
    """Raised when the application can't proceed with a user request
    for a model version update because the model it's already final"""
