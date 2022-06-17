import pytest
from subprocess import check_output, CalledProcessError
from app.features.model import generate


def test_generate_bundle():
    python_code = generate.generate_bundle()
    assert isinstance(python_code, str)
    try:
        check_output(['python'], input=python_code, text=True)
    except CalledProcessError:
        pytest.fail("Failed to exectute generated python bundle`")

