# Import all the models, so that Base has them before being
# imported by Alembic
from app.db.base_class import Base  # noqa
from app.features.user.model import User  # noqa
from app.features.dataset.model import Dataset# noqa
