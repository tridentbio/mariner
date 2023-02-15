import pytest
from pydantic import ValidationError

from mariner.nn_validation import CheckTypeHints


class TestCheckTypeHints:
    def test_valid_types_list(self):
        CheckTypeHints(
            types=["typing.List[str]", "list[str]"], expected_type="list[str]"
        )

    def test_valid_optionals(self):
        # strings ending in ? are
        CheckTypeHints(
            types=["<class 'torch.Tensor'>?"],
            expected_type="typing.Optional[torch.Tensor]",
        )
        # None is Optional
        CheckTypeHints(types=["None"], expected_type="typing.Optional[torch.Tensor]")

    def test_valid_unions(self):
        # None belongs to Union[None, ...]
        CheckTypeHints(types=["None"], expected_type="typing.Union[None, torch.Tensor]")

    def test_raises_on_invalid(self):
        # Dict is not list
        with pytest.raises(ValidationError):
            CheckTypeHints(types=["typing.Dict"], expected_type="list")

    def test_invalid_with_class(self):
        # Tensor is not list
        with pytest.raises(ValidationError):
            CheckTypeHints(types=["<class 'torch.Tensor'>"], expected_type="list")

    def test_valid_types_json(self):
        CheckTypeHints.parse_raw(
            """
            {
                "types": ["list[str]", "typing.List[str]"],
                "expectedType": "list[str]"
            }"""
        )

    def test_raises_on_invalid_json(self):
        with pytest.raises(ValidationError):
            CheckTypeHints.parse_raw(
                """
                {
                    "types": ["typing.Dict"],
                    "expectedType": "list"
                }"""
            )
