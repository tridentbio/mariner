import pytest
import torch
from sqlalchemy.orm.session import Session

from mariner import models as model_ctl
from mariner.entities import Dataset as DatasetEntity
from mariner.entities import Model as ModelEntity
from mariner.schemas.model_schemas import Model, TrainingCheckRequest
from mariner.stores.dataset_sql import dataset_store
from tests.fixtures.model import model_config
from tests.fixtures.user import get_test_user


@pytest.mark.asyncio
@pytest.mark.integration
async def test_get_model_prediction(db: Session, some_trained_model: Model):
    version = some_trained_model.versions[-1]
    test_user = get_test_user(db)
    ds = dataset_store.get(db, some_trained_model.dataset_id)
    assert ds
    df = ds.get_dataframe()
    df = df.to_dict()
    assert df
    result = model_ctl.get_model_prediction(
        db,
        model_ctl.PredictRequest(
            user_id=test_user.id, model_version_id=version.id, model_input=df
        ),
    )
    for prediction in result.values():
        assert isinstance(prediction, torch.Tensor)


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
    check = await model_ctl.check_model_step_exception(
        db, TrainingCheckRequest(model_spec=regressor)
    )
    assert regressor.dataset.target_columns[0].loss_fn
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
