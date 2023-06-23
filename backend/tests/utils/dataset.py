import json
from typing import Dict, List

from mariner.schemas.dataset_schemas import ColumnsMeta

from .utils import random_lower_string


def get_post_dataset_data(metadata: List[ColumnsMeta] = None) -> Dict[str, str]:
    metadatas_example: List[ColumnsMeta] = metadata or [
        {
            "pattern": "exp",
            "data_type": {"domain_kind": "numeric", "unit": "mole"},
            "description": "experiment measurement",
            "unit": "mole",
        },
        {
            "pattern": "smiles",
            "data_type": {
                "domain_kind": "smiles",
            },
            "description": "SMILES representaion of molecule",
        },
    ]

    return {
        "name": random_lower_string(),
        "description": "Test description",
        "splitType": "random",
        "splitTarget": "60-20-20",
        "columnsMetadata": json.dumps(metadatas_example),
    }
