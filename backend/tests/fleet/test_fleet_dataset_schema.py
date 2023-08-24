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
        assert isinstance(spec.dataset, DatasetConfigWithPreprocessing), (
            "Expected spec.dataset to be DatasetConfigWithPreprocessing "
            f"got {type(spec.dataset)}"
        )
        dataset_config = spec.dataset.to_dataset_config()
        assert dataset_config
        assert dataset_config.featurizers[0].name == "smiles-out", (
            'Expected first featurizer to be "smiles-out", '
            f"got {dataset_config.featurizers[0].name}"
        )
        assert (
            dataset_config.featurizers[0].type
            == "molfeat.trans.fp.FPVecFilteredTransformer"
        )

    with helpers.load_test(
        "sklearn_sampl_knearest_neighbor_regressor.yaml"
    ) as f:
        assert isinstance(f, dict)
        spec = Foo.parse_obj({"spec": f}).spec
        assert isinstance(spec.dataset, DatasetConfigWithPreprocessing), (
            "Expected spec.dataset to be DatasetConfigWithPreprocessing "
            f"got {type(spec.dataset)}"
        )
        dataset_config = spec.dataset.to_dataset_config()
        assert dataset_config
        assert dataset_config.featurizers[0].name == "smiles-out", (
            'Expected first featurizer to be "smiles-out", '
            f"got {dataset_config.featurizers[0].name}"
        )
        assert (
            dataset_config.featurizers[0].type
            == "molfeat.trans.fp.FPVecFilteredTransformer"
        )
