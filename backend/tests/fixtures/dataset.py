import json
from datetime import datetime
from typing import Dict, Literal, Optional

from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from mariner.core import aws
from mariner.core.config import get_app_settings
from mariner.entities.dataset import Dataset
from mariner.schemas.dataset_schemas import DatasetCreateRepo
from mariner.stores import dataset_sql
from tests.fixtures.user import get_test_user


def mock_columns_metadatas(dataset: Literal["zinc", "sampl", "hiv"] = "zinc"):
    if dataset == "zinc":
        return [
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
    elif dataset == "sampl":
        return [
            {
                "dataType": {"domainKind": "string"},
                "description": "column 1",
                "pattern": "iupac",
            },
            {
                "dataType": {"domainKind": "smiles"},
                "description": "column 2",
                "pattern": "smiles",
            },
            {
                "dataType": {"domainKind": "numeric", "unit": "mole"},
                "description": "column 3",
                "pattern": "expt",
            },
            {
                "dataType": {"domainKind": "numeric", "unit": "mole"},
                "description": "column 4",
                "pattern": "calc",
            },
            {
                "dataType": {
                    "domainKind": "categorical",
                    "classes": {"1": 0, "2": 1, "3": 2},
                },
                "description": "column ",
                "pattern": "step",
            },
        ]
    elif dataset == "hiv":
        return [
            {
                "dataType": {"domainKind": "smiles"},
                "description": "",
                "pattern": "smiles",
            },
            {
                "dataType": {
                    "domainKind": "categorical",
                    "classes": {"CI": 0, "CM": 1},
                },
                "description": "",
                "pattern": "activity",
            },
            {
                "dataType": {
                    "domainKind": "categorical",
                    "classes": {"0": 0, "1": 1},
                },
                "description": "",
                "pattern": "HIV_active",
            },
            {
                "dataType": {
                    "domainKind": "categorical",
                    "classes": {"1": 0, "2": 1, "3": 2},
                },
                "description": "",
                "pattern": "step",
            },
        ]


def mock_dataset(
    name: Optional[str] = None,
    file: str = "tests/data/csv/zinc_extra.csv",
    dataset: Literal["zinc", "sampl", "hiv"] = "zinc",
):
    key: str = f"datasets/{dataset}.csv"
    if not aws.is_in_s3(key=key, bucket=aws.Bucket.Datasets):
        with open(file, "rb") as f:
            aws.upload_s3_file(f, key=key, bucket=aws.Bucket.Datasets)

    return {
        "name": name if name else "Small Zinc dataset",
        "description": "Test description",
        "splitType": "random",
        "splitTarget": "60-20-20",
        "dataUrl": key,
        "columnsMetadata": json.dumps(mock_columns_metadatas(dataset)),
    }


def setup_create_dataset(
    db: Session,
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    file="tests/data/csv/zinc_extra.csv",
    **kwargs,
):
    data = mock_dataset(**kwargs)
    db.query(Dataset).filter(Dataset.name == data["name"]).delete()
    db.commit()
    with open(file, "rb") as f:
        res = client.post(
            f"{get_app_settings().API_V1_STR}/datasets/",
            data=data,
            files={"file": ("zinc_extra.csv", f.read())},
            headers=normal_user_token_headers,
        )
        assert (
            res.status_code == status.HTTP_200_OK
        ), "dataset fixture creation failed"
        body = res.json()
        assert body["id"] >= 1, "dataset fixture creation failed"
        return db.query(Dataset).get(body["id"])


def teardown_create_dataset(db: Session, dataset: Dataset):
    ds = db.query(Dataset).get(dataset.id)
    assert ds is not None
    db.delete(ds)
    db.commit()


def setup_create_dataset_db(
    db: Session,
    **kwargs,
):
    data = mock_dataset(**kwargs)
    user = get_test_user(db)
    db.query(Dataset).filter(Dataset.name == data["name"]).delete()
    data.pop("columnsMetadata")
    create_data = DatasetCreateRepo(
        **data,
        updated_at=datetime.now(),
        created_at=datetime.now(),
        created_by_id=user.id,
        columns_metadata=mock_columns_metadatas(),
        from_alias=True,
    )
    dataset = dataset_sql.dataset_store.create(db, create_data)
    db.commit()
    return dataset
