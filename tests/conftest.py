import datetime
import mlflow.tracking
import json
from typing import Dict, Generator, List, Literal, Optional
from mlflow import mlflow

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from starlette import status
import yaml

from mariner.core.aws import Bucket, delete_s3_file
from mariner.core.config import settings
from mariner.db.session import SessionLocal
from mariner.entities import (
    Dataset,
    Experiment as ExperimentEntity,
    User,
    Model as ModelEntity,
    EventEntity,
)
from mariner.schemas.experiment_schemas import Experiment
from mariner.stores.event_sql import EventCreateRepo, event_store
from model_builder.schemas import ModelSchema
from mariner.schemas.model_schemas import Model, ModelCreate, ModelVersion
from mariner.stores.user_sql import user_store
from mariner.stores.experiment_sql import ExperimentCreateRepo, experiment_store
from api.fastapi_app import app
from tests.utils.user import authentication_token_from_email
from tests.utils.utils import (
    get_superuser_token_headers,
    random_lower_string,
)


@pytest.fixture(scope="session")
def db() -> Generator:
    sesion = SessionLocal()
    yield sesion


@pytest.fixture(scope="module")
def client() -> Generator:
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def superuser_token_headers(client: TestClient) -> Dict[str, str]:
    return get_superuser_token_headers(client)


@pytest.fixture(scope="module")
def normal_user_token_headers(client: TestClient, db: Session) -> Dict[str, str]:
    return authentication_token_from_email(
        client=client, email=settings.EMAIL_TEST_USER, db=db
    )


def get_test_user(db: Session) -> User:
    user = user_store.get_by_email(db, email=settings.EMAIL_TEST_USER)
    assert user is not None
    return user


# DATASET GLOBAL FIXTURES
def mock_dataset(name: Optional[str] = None):
    metadatas = [
        {
            "pattern": "smiles",
            "data_type": {
                "domain_kind": "smiles",
            },
            "description": "smiles column",
        },
        {
            "pattern": "mwt",
            "data_type": {"domain_kind": "numeric", "unit": "mole"},
            "description": "Molecular Weigth",
        },
        {
            "pattern": "tpsa",
            "data_type": {"domain_kind": "numeric", "unit": "mole"},
            "description": "T Polar surface",
        },
        {
            "pattern": "mwt_group",
            "data_type": {
                "domain_kind": "categorical",
                "classes": {"yes": 0, "no": 1},
            },
            "description": "yes if mwt is larger than 300 otherwise no",
        },
    ]

    return {
        "name": name if name else "Small Zinc dataset",
        "description": "Test description",
        "splitType": "random",
        "splitTarget": "60-20-20",
        "columnsMetadata": json.dumps(metadatas),
    }


def setup_create_dataset(
    db: Session, client: TestClient, normal_user_token_headers: Dict[str, str]
):
    data = mock_dataset()
    db.query(Dataset).filter(Dataset.name == data["name"]).delete()
    db.commit()
    with open("tests/data/zinc_extra.csv", "rb") as f:
        res = client.post(
            f"{settings.API_V1_STR}/datasets/",
            data=data,
            files={"file": ("zinc_extra.csv", f.read())},
            headers=normal_user_token_headers,
        )
        assert res.status_code == status.HTTP_200_OK
        body = res.json()
        return db.query(Dataset).get(body["id"])


def teardown_create_dataset(db: Session, dataset: Dataset):
    ds = db.query(Dataset).get(dataset.id)
    assert ds is not None
    db.delete(ds)
    db.commit()
    try:
        delete_s3_file(bucket=Bucket.Datasets, key=ds.data_url)
    except Exception:
        pass


@pytest.fixture(scope="module")
def some_dataset(
    db: Session, client: TestClient, normal_user_token_headers: Dict[str, str]
):
    ds = setup_create_dataset(db, client, normal_user_token_headers)
    assert ds is not None
    yield ds
    # teardown_create_dataset(db, ds)


# MODEL GLOBAL FIXTURES


@pytest.fixture(scope="module")
def some_model(
    db: Session,
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    some_dataset: Dataset,
):
    model = setup_create_model(client, normal_user_token_headers, some_dataset)
    yield model
    teardown_create_model(db, model)


def mock_dataset_item():
    import torch
    from torch_geometric.data import Data

    x = torch.ones(21, 26, dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    mwt = torch.tensor([[230.0]], dtype=torch.float)

    dataset_input = {
        "MolToGraphFeaturizer": Data(x=x, edge_index=edge_index, batch=None),
        "mwt": mwt,
        "tpsa": mwt,
    }

    return dataset_input


@pytest.fixture(scope="module")
def dataset_sample():
    return mock_dataset_item()


@pytest.fixture(scope="module")
def model_config() -> Generator[ModelSchema, None, None]:
    path = "tests/data/test_model_hard.yaml"
    with open(path, "rb") as f:
        yield ModelSchema.from_yaml(f.read())


def mock_experiment(
    version: ModelVersion,
    user_id: int,
    stage: Optional[Literal["started", "success"]] = None,
):
    create_obj = ExperimentCreateRepo(
        epochs=1,
        experiment_name=random_lower_string(),
        mlflow_id=random_lower_string(),
        created_by_id=user_id,
        model_version_id=version.id,
    )
    if stage == "started":
        pass  # create_obj is ready
    elif stage == "success":
        create_obj.history = {
            "train_loss": [300.3, 210.9, 160.8, 130.3, 80.4, 50.1, 20.0]
        }
        create_obj.train_metrics = {"train_loss": 200.3}
        create_obj.stage = "SUCCESS"
    else:
        raise NotImplementedError()
    return create_obj


def mock_model(name=None, dataset_name=None) -> ModelCreate:
    model_path = "tests/data/test_model_hard.yaml"
    with open(model_path, "rb") as f:
        config_dict = yaml.unsafe_load(f.read())
        config = ModelSchema.parse_obj(config_dict)
        if dataset_name:
            config.dataset.name = dataset_name
        model = ModelCreate(
            name=name if name is not None else random_lower_string(),
            model_description=random_lower_string(),
            model_version_description=random_lower_string(),
            config=config,
        )
        return model


def setup_create_model(client: TestClient, headers, dataset: Optional[Dataset] = None):
    model = None
    if dataset:
        model = mock_model(dataset_name=dataset.name)
    else:
        model = mock_model()
    data = model.dict()
    res = client.post(
        f"{settings.API_V1_STR}/models/",
        json=data,
        headers=headers,
    )
    assert res.status_code == status.HTTP_200_OK
    return Model.parse_obj(res.json())


def teardown_create_model(db: Session, model: Model):
    obj = db.query(ModelVersion).filter(ModelVersion.model_id == model.id).first()
    db.delete(obj)
    obj = db.query(ModelEntity).filter(ModelEntity.id == model.id).first()
    if obj:
        db.delete(obj)
        db.flush()
        mlflowclient = mlflow.tracking.MlflowClient()
        mlflowclient.delete_registered_model(model.mlflow_name)


@pytest.fixture(scope="function")
def model(
    db: Session,
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    some_dataset: Dataset,
):
    db.commit()
    model = setup_create_model(client, normal_user_token_headers, some_dataset)
    yield model
    dbobj = db.query(ModelEntity).filter(ModelEntity.id == model.id).first()
    if dbobj:
        db.delete(dbobj)
        db.flush()


def get_test_events(
    db: Session, some_experiments: List[Experiment]
) -> List[EventEntity]:
    user = get_test_user(db)
    mocked_events = [
        EventCreateRepo(
            user_id=user.id,
            source="training:completed",
            timestamp=datetime.datetime(2022, 10, 29),
            payload=experiment.dict(),
            url="",
        )
        for experiment in some_experiments
    ]
    events = [event_store.create(db, obj_in=mock) for mock in mocked_events]
    return events


def teardown_events(db: Session, events: List[EventEntity]):
    db.commit()
    for event in events:
        db.delete(event)
    db.flush()


@pytest.fixture(scope="module")
def events_fixture(db: Session, some_experiments: List[Experiment]):
    user = get_test_user(db)
    db.query(EventEntity).filter(EventEntity.user_id == user.id).delete()
    db.flush()
    events = get_test_events(db, some_experiments)
    yield events
    teardown_events(db, events)


@pytest.fixture(scope="function")
def some_experiment(
    db: Session, some_model: Model
) -> Generator[Experiment, None, None]:
    user = get_test_user(db)
    version = some_model.versions[-1]
    exp = experiment_store.create(
        db, obj_in=mock_experiment(version, user.id, stage="started")
    )
    assert exp
    exp = Experiment.from_orm(exp)
    yield exp
    db.query(ExperimentEntity).filter(ExperimentEntity.id == exp.id).delete()


@pytest.fixture(scope="function")
def some_cmoplete_experiment(
    db: Session, some_model: Model
) -> Generator[Experiment, None, None]:
    user = get_test_user(db)
    version = some_model.versions[-1]
    exp = experiment_store.create(
        db, obj_in=mock_experiment(version, user.id, stage="success")
    )
    assert exp
    exp = Experiment.from_orm(exp)
    yield exp
    db.query(ExperimentEntity).filter(ExperimentEntity.id == exp.id).delete()
    db.commit()


@pytest.fixture(scope="function")
def some_experiments(
    db: Session, some_model: Model
) -> Generator[List[Experiment], None, None]:
    user = get_test_user(db)
    version = some_model.versions[-1]

    user = get_test_user(db)
    version = some_model.versions[-1]
    exps = [
        experiment_store.create(
            db, obj_in=mock_experiment(version, user.id, stage="started")
        )
        for _ in range(0, 3)
    ]
    exps = [Experiment.from_orm(exp) for exp in exps]
    yield exps
    db.query(ExperimentEntity).filter(
        ExperimentEntity.id.in_([exp.id for exp in exps])
    ).delete()
    db.flush()
