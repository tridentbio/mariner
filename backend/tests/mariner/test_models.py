from datetime import datetime
from pathlib import Path

import pytest
import ray
from botocore.endpoint import time
from sqlalchemy.orm.session import Session

from fleet.base_schemas import TorchModelSpec
from fleet.ray_actors.tasks import get_task_control
from fleet.utils.dataset import converts_file_to_dataframe
from mariner import models as model_ctl
from mariner.entities import Dataset as DatasetEntity
from mariner.entities import Model as ModelEntity
from mariner.entities.model import ModelVersion
from mariner.schemas.dataset_schemas import Dataset as DatasetSchema
from mariner.schemas.model_schemas import Model, ModelCreate
from mariner.stores.dataset_sql import dataset_store
from tests.fixtures.model import model_config
from tests.fixtures.user import get_test_user


@pytest.mark.asyncio
@pytest.mark.integration
async def test_get_model_prediction(db: Session, some_trained_model: Model):
    version = some_trained_model.versions[-1]
    test_user = get_test_user(db)
    ds = DatasetSchema.from_orm(
        dataset_store.get(db, some_trained_model.dataset_id)
    )
    assert ds
    df = converts_file_to_dataframe(ds.get_dataset_file())
    df = df.to_dict()
    assert df
    result = model_ctl.get_model_prediction(
        db,
        model_ctl.PredictRequest(
            user_id=test_user.id, model_version_id=version.id, model_input=df
        ),
    )
    for prediction in result.values():
        assert isinstance(prediction, list)


@pytest.mark.integration
def test_delete_model(db: Session, model: Model):
    user = get_test_user(db)
    model_ctl.delete_model(db, user, model.id)
    model_db = (
        db.query(ModelEntity).filter(ModelEntity.name == model.name).first()
    )
    assert not model_db


@pytest.mark.asyncio
@pytest.mark.integration
async def test_check_forward_exception_good_regressor(
    db: Session, some_dataset: DatasetEntity
):
    regressor = model_config(
        model_type="regressor", dataset_name=some_dataset.name
    )
    assert regressor.dataset.target_columns[0].loss_fn
    user = get_test_user(db)
    model_version = ModelVersion(
        id=1,
        model_id=1,
        description="askdas",
        name="iajsdijiasd",
        mlflow_version="iasjdijasda",
        mlflow_model_name="asjdkasjdk",
        config=regressor,
        created_by_id=user.id,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    check = model_ctl.start_check_model_step_exception(
        model_version=model_version, user=user, task_control=get_task_control()
    )
    check = ray.get(check)
    assert check.stack_trace is None, check.stack_trace
    assert check.output is not None


def test_get_model_options():
    # Should return a list of ComponentOption
    options = model_ctl.get_model_options()
    assert len(options) > 0
    for option in options:
        assert option.type
        if option.class_path == "torch.nn.TransformerEncoderLayer":
            assert option.args_options
            assert "activation" in option.args_options
            assert len(option.args_options["activation"]) > 0


@pytest.mark.asyncio
async def test_update_model_version(db: Session, some_dataset: DatasetSchema):
    # Create test model that fails check
    user = get_test_user(db)
    import yaml

    with open(
        Path("tests") / "data" / "yaml" / "small_regressor_schema.yaml"
    ) as config_file:
        config_dict = yaml.unsafe_load(config_file)
    config = TorchModelSpec.parse_obj(config_dict)
    config.spec.layers[0].constructor_args.in_channels = 1
    model = await model_ctl.create_model(
        db=db,
        user=user,
        model_create=ModelCreate(
            name="Test model",
            model_description="This is a model description",
            model_version_description="This is a model version description",
            config=config,
        ),
    )
    assert model.name == "Test model"
    assert model.versions[0].check_status == None
    ids, tasks = get_task_control().get_tasks(
        {
            "model_version_id": model.versions[0].id,
        }
    )
    assert len(tasks) == len(ids) == 1
    ray.get(tasks[0])
    time.sleep(5)
    modelversion = (
        db.query(ModelVersion)
        .filter(ModelVersion.id == model.versions[0].id)
        .first()
    )
    assert modelversion.check_status == "FAILED"
    # Update model version with new config
