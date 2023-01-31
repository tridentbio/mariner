import json
from typing import Dict, Optional

from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from mariner.core.config import settings
from mariner.entities.dataset import Dataset


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
    db: Session,
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    file="tests/data/zinc_extra.csv",
):
    data = mock_dataset()
    db.query(Dataset).filter(Dataset.name == data["name"]).delete()
    db.commit()
    with open(file, "rb") as f:
        res = client.post(
            f"{settings.API_V1_STR}/datasets/",
            data=data,
            files={"file": ("zinc_extra.csv", f.read())},
            headers=normal_user_token_headers,
        )
        assert res.status_code == status.HTTP_200_OK, "dataset fixture creation failed"
        body = res.json()
        assert body["id"] >= 1, "dataset fixture creation failed"
        return db.query(Dataset).get(body["id"])


def teardown_create_dataset(db: Session, dataset: Dataset):
    ds = db.query(Dataset).get(dataset.id)
    assert ds is not None
    db.delete(ds)
    db.commit()
