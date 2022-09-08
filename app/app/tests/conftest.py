import json
from typing import Dict, Generator, Literal, Optional

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from starlette import status

from app.core.aws import Bucket, delete_s3_file
from app.core.config import settings
from app.db.session import SessionLocal
from app.features.dataset.model import Dataset
from app.features.experiments.crud import ExperimentCreateRepo
from app.features.experiments.crud import repo as experiments_repo
from app.features.experiments.model import Experiment as ExperimentEntity
from app.features.experiments.schema import Experiment
from app.features.model.schema.configs import ModelConfig
from app.features.model.schema.model import Model, ModelVersion
from app.features.user.crud import repo as user_repo
from app.features.user.model import User
from app.main import app
from app.tests.features.model.conftest import (
    setup_create_model,
    teardown_create_model,
)
from app.tests.utils.user import authentication_token_from_email
from app.tests.utils.utils import (
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
    user = user_repo.get_by_email(db, email=settings.EMAIL_TEST_USER)
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
    with open("app/tests/data/zinc_extra.csv", "rb") as f:
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
def model_config() -> Generator[ModelConfig, None, None]:
    path = "app/tests/data/test_model_hard.yaml"
    with open(path, "rb") as f:
        yield ModelConfig.from_yaml(f.read())


def mock_experiment(
    version: ModelVersion,
    user_id: int,
    stage: Optional[Literal["started", "success"]] = None,
):
    create_obj = ExperimentCreateRepo(
        epochs=1,
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


@pytest.fixture(scope="module")
def some_experiments(db, some_model: Model):
    db.commit()
    user = get_test_user(db)
    version = some_model.versions[-1]
    exps = [
        Experiment.from_orm(
            experiments_repo.create(
                db, obj_in=mock_experiment(version, user.id, stage="started")
            )
        )
        for _ in range(3)
    ]
    yield exps
    ids = [exp.id for exp in exps]
    db.query(ExperimentEntity).filter(ExperimentEntity.id.in_(ids)).delete()
    db.flush()
