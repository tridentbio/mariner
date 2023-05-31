"""
This file contains some utility functions that are used in the tests.
"""
import random
import string
from typing import Iterator


def random_lower_string() -> str:
    """
    Generate a random string of 32 characters.
    """
    return "".join(random.choices(string.ascii_lowercase, k=32))


def random_email() -> str:
    """
    Generate a random email.
    """
    return f"{random_lower_string()}@{random_lower_string()}.com"


def assert_all_is_number(arr: Iterator):
    """
    Assert that all values in the list are numbers.
    """
    for item in arr:
        assert isinstance(item, (int, float)), f"Value {item} is not a number"
