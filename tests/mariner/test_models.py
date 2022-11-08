from typing import Any

import torch
from mockito import patch
from sqlalchemy.orm.session import Session

from mariner import models as model_ctl
from mariner.entities import Dataset as DatasetEntity
from mariner.entities import Model as ModelEntity
from mariner.schemas.model_schemas import Model
from mariner.stores.dataset_sql import dataset_store
from model_builder.schemas import ModelSchema
from tests.conftest import get_test_user


def test_get_model_prediction(db: Session, model: Model):
    version = model.versions[-1]
    test_user = get_test_user(db)
    ds = dataset_store.get(db, model.dataset_id)
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
    for tpsa in result:
        assert isinstance(tpsa, torch.Tensor)


def test_delete_model(db: Session, model: Model):
    user = get_test_user(db)
    model_ctl.delete_model(db, user, model.id)
    model_db = db.query(ModelEntity).filter(ModelEntity.name == model.name).first()
    assert not model_db


def test_check_forward_exception_good_model(
    db: Session, some_dataset: DatasetEntity, model_config: ModelSchema
):
    check = model_ctl.naive_check_forward_exception(db, model_config)
    assert check.stack_trace is None
    assert check.output is not None


def test_check_forward_exception_bad_model(db: Session, model_config: ModelSchema):
    import model_builder.model

    def raise_(x: Any):
        raise Exception("bad bad model")

    with patch(model_builder.model.CustomModel.forward, raise_):
        check = model_ctl.naive_check_forward_exception(db, model_config)
        assert check.output is None
        assert check.stack_trace is not None
