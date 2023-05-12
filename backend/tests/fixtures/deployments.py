from mariner.models import Model

from ..utils.utils import random_lower_string


def mock_deployment(some_model: Model, share_strategy="private"):
    return {
        "name": random_lower_string(),
        "readme": random_lower_string(),
        "status": "stopped",
        "model_version_id": some_model.versions[0].id,
        "share_strategy": share_strategy,
        "prediction_rate_limit_value": 100,
        "prediction_rate_limit_unit": "month",
    }
