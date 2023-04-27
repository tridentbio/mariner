"""
Exceptions of the model_builder package
"""
from typing import Any, Union


class DataTypeMismatchException(Exception):
    """
    Exception to be raised when a operation cannot proceed due mismatching data types
    """

    def __init__(self, msg: str, expected: Union[type, None], got_item: Any):
        self.expected = expected
        self.got_item = got_item
        self.msg = msg
        super().__init__(msg)
