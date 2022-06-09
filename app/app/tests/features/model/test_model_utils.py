from dataclasses import dataclass
from typing import Optional

from app.features.model.utils import get_inputs_from_mask_ltr


@dataclass
class GetInputsFromMaskTestCase:
    args: tuple[list, int]
    expected_len: Optional[int]
    expected: Optional[list]


def test_get_inputs_from_mask_lrt():
    cases = [
        GetInputsFromMaskTestCase(
            args=(["a", "b", "c"], 0b100), expected_len=1, expected=["a"]
        ),
        GetInputsFromMaskTestCase(
            args=(["a", "b", "c"], 0b101), expected_len=2, expected=["a", "c"]
        ),
        GetInputsFromMaskTestCase(
            args=(["a", "b", "c"], 0b011), expected_len=2, expected=["b", "c"]
        ),
    ]

    for test_case in cases:
        arr = get_inputs_from_mask_ltr(*test_case.args)
        if test_case.expected_len:
            assert len(arr) == test_case.expected_len
        if test_case.expected:
            assert arr == test_case.expected
