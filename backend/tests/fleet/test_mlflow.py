import mlflow
import pytest
import yaml

from fleet.dataset_schemas import DatasetConfig
from fleet.mlflow import load_pipeline, save_pipeline
from fleet.utils.data import PreprocessingPipeline


@pytest.fixture(scope="module")
def preprocessing_pipeline_fixture():
    yaml_path = "tests/data/yaml/sklearn_hiv_extra_trees_classifier.yaml"
    model_spec = yaml.safe_load(open(yaml_path, "r"))
    dataset_spec = model_spec["dataset"]
    return DatasetConfig.parse_obj(dataset_spec)


def test_save_load_pipeline(preprocessing_pipeline_fixture: DatasetConfig):
    run_id = None
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        pipeline = PreprocessingPipeline(
            dataset_config=preprocessing_pipeline_fixture
        )
        save_pipeline(pipeline)

    loaded_pipeline = load_pipeline(run_id)

    assert isinstance(loaded_pipeline, PreprocessingPipeline)
    assert len(pipeline.featurizers) == len(loaded_pipeline.featurizers)
    assert len(pipeline.transforms) == len(loaded_pipeline.transforms)
