"""Migrate model_version config property to new TorchModelSpec

Revision ID: 40fadbde09b7
Revises: 809377f238f8
Create Date: 2023-05-01 16:48:44.960040

"""

from sqlalchemy.exc import ProgrammingError

from mariner.db.session import SessionLocal
from mariner.entities.model import ModelVersion

from fleet.torch_.schemas import TorchModelSpec

# revision identifiers, used by Alembic.
revision = "40fadbde09b7"
down_revision = "809377f238f8"
branch_labels = None
depends_on = None


def convert_to_spec(old_model_schema: dict) -> "TorchModelSpec":
    new_spec = {}
    new_spec["framework"] = "torch"
    new_spec["name"] = old_model_schema["name"]
    new_spec["dataset"] = old_model_schema["dataset"]
    new_spec["dataset"]["featurizers"] = old_model_schema["featurizers"]
    new_spec["spec"] = {"layers": old_model_schema["layers"]}

    for layer in new_spec["spec"]["layers"]:
        if layer["type"].startswith("model_builder"):
            layer["type"] = f"fleet.{layer['type']}"

    for feat in new_spec["dataset"]["featurizers"]:
        if feat["type"].startswith("model_builder"):
            feat["type"] = f"fleet.{feat['type']}"

    return TorchModelSpec(**new_spec)


def undo_convert_to_spec(spec: dict) -> dict:
    new_spec = {}
    new_spec["name"] = spec["name"]
    new_spec["dataset"] = spec["dataset"]
    new_spec["featurizers"] = spec["dataset"]["featurizers"]
    new_spec["layers"] = spec["spec"]["layers"]

    for layer in new_spec["layers"]:
        if layer["type"].startswith("fleet.model_builder"):
            layer["type"] = layer["type"].replace(
                "fleet.model_builder", "model_builder"
            )

    for feat in new_spec["featurizers"]:
        if feat["type"].startswith("fleet.model_builder"):
            feat["type"] = feat["type"].replace("fleet.model_builder", "model_builder")

    return new_spec


def upgrade():
    with SessionLocal() as db:
        try:
            model_versions = db.query(ModelVersion).all()
            for model_version in model_versions:
                spec = convert_to_spec(model_version.config)
                model_version.config = spec.dict()
                db.commit()
                db.flush()
        except ProgrammingError as exp:
            if exp.code == "42P01":  # Undefined Table
                print("Table is being created in current upgrade")
                print("No model version specs to migrate")


def downgrade():
    with SessionLocal() as db:
        model_versions = db.query(ModelVersion).all()
        for model_version in model_versions:
            spec = undo_convert_to_spec(model_version.config)
            model_version.config = spec
            db.commit()
            db.flush()
