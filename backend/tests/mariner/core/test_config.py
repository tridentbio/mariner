"""
Tests the mariner.core.config module, responsible for loading and
validating the app configuration.
"""

import os
import re
from dataclasses import dataclass
from typing import Union
from unittest import mock

import pytest
from pydantic import ValidationError

from mariner.core.config import get_app_settings


@dataclass
class TestCase:
    """
    A test case for the Config class.
    """

    environment: Union[dict, None] = None
    raises: Union[type, None] = None
    raises_regex: Union[re.Pattern, None] = None
    environment_extend: bool = True


def build_contains_regex_pattern(strings):
    # Escape special characters in the strings to avoid regex errors
    escaped_strings = [re.escape(s) for s in strings]

    # Construct the regex pattern using lookahead assertions
    pattern = r"(?=.*" + r")(?=.*".join(escaped_strings) + r")"

    # Compile the regex pattern into a re.Pattern object
    compiled_pattern = re.compile(pattern)

    return compiled_pattern


test_cases = [
    # Uses default environment loaded by pytest
    TestCase(),
    # A case with missing environment variables.
    TestCase(environment={}, raises=ValidationError, environment_extend=False),
    # A case with incomplete oauth variables.
    TestCase(
        environment={"OAUTH_PROV_CLIENT_ID": "test"},
        raises=ValueError,
        # The error message should contain the missing variables:
        # OAUTH_PROV_CLIENT_SECRET, OAUTH_PROV_NAME
        raises_regex=build_contains_regex_pattern(
            [
                "OAUTH_PROV_CLIENT_SECRET",
                "OAUTH_PROV_NAME",
                "OAUTH_PROV_AUTHORIZATION_URL",
                "OAUTH_PROV_SCOPE",
            ]
        ),
    ),
    TestCase(
        environment={
            "OAUTH_PROV_CLIENT_ID": "test",
            "OAUTH_PROV_CLIENT_SECRET": "test",
            "OAUTH_PROV_CLIENT_COPE": "test",
        },
        raises=ValueError,
        raises_regex=build_contains_regex_pattern(
            [
                "OAUTH_PROV_NAME",
                "OAUTH_PROV_AUTHORIZATION_URL",
            ]
        ),
    ),
]


@pytest.mark.first
@pytest.mark.parametrize("case", test_cases)
def test_get_app_settings(case):
    """
    Tests the Config class.
    """
    if case.environment is None:
        assert (
            case.raises is None
        ), "pytest default environment shouldn't raise errors"
        settings = get_app_settings(use_cache=False)
        assert settings is not None
        return

    with mock.patch.dict(
        os.environ, case.environment or {}, clear=not case.environment_extend
    ):
        if case.raises is not None:
            try:
                # print aws variables for debugging
                settings = get_app_settings(use_cache=False)
                pytest.fail("Should have raised an error")
            except case.raises as exc:
                # Checks if message matches the regex pattern
                if case.raises_regex is not None:
                    thing = case.raises_regex.search(str(exc))
                    assert (
                        thing is not None
                    ), f"Error message {str(exc)} does not match regex pattern "
        else:
            get_app_settings(use_cache=False)
