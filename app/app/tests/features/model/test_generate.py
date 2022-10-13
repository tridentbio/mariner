from subprocess import CalledProcessError, check_output

import pytest

from app.features.model import generate


def test_generate_bundle():
    python_code = generate.generate_bundle()
    assert isinstance(python_code, str)
    try:
        check_output(["python"], input=python_code, text=True)
    except CalledProcessError:
        pytest.fail("Failed to exectute generated python bundle`")


# TODO:
# Parameterized test for some layers and featurizer classes
# where we check the _get_component_signature properly gets
# and separates the arguments for each of the test cases
# - Linear
# - ReLU
# - GCN
# - Some aggregation layer from pygnn
# - Some pooling layer from pygnn
# - MoleculeFeaturizer
# - Concat
def test_get_component_signature():
    ...
