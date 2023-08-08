from pydantic import BaseModel

from fleet.base_schemas import FleetModelSpec
from fleet.dataset_schemas import DatasetConfigWithPreprocessing
from tests.fleet import helpers


def test_sklearn_spec_supports_DatasetConfigWithPreprocessing():
    class Foo(BaseModel):
        spec: FleetModelSpec

    with helpers.load_test(
        "sklearn_sampl_knearest_neighbor_regressor.yaml"
    ) as f:
        assert isinstance(f, dict)
        spec = Foo.parse_obj({"spec": f}).spec
        assert isinstance(
            spec.dataset, DatasetConfigWithPreprocessing
        ), "DatasetConfigWithPreprocessing not supported by sklearn models"
