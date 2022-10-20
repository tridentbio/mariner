from app.features.model.schema.model import Model


class TestModelVersion:
    """
    Test suite for the app.features.model.schema.ModelVersion class
    """

    def test_load_mlflow_entity(self, model: Model):
        assert len(model.versions)
        version = model.versions[0]
        assert version.mlflow_model_name == model.mlflow_name
        mlflow_version_entity = version.load_from_mlflowapi()
        assert mlflow_version_entity
        assert mlflow_version_entity.version == version.mlflow_version
        assert mlflow_version_entity.name == version.mlflow_model_name
