import random
import string
from typing import Iterator


def random_lower_string() -> str:
    return "".join(random.choices(string.ascii_lowercase, k=32))


def random_email() -> str:
    return f"{random_lower_string()}@{random_lower_string()}.com"


def assert_all_is_number(l: Iterator):
    for v in l:
        assert isinstance(v, (int, float)), f"Value {v} is not a number"
