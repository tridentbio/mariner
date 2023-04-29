from mariner.models import Model
from ..utils.utils import random_lower_string


def mock_deploy(some_model: Model, share_strategy='private'):
    return {
        "name": random_lower_string(),
        "readme": random_lower_string(),
        "status": 'stopped',
        "model_version_id": some_model.versions[0].id,
        "share_strategy": share_strategy,
        "rate_limit_value": 100,
        "rate_limit_unit": "month",
    }