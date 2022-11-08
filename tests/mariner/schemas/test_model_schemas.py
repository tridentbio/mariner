from mariner.schemas.model_schemas import Model


class TestModel:
    """
    Test suite for the app.features.model.schema.model
    class
    """

    def test_load_mlflow_entity(self, model: Model):
        mlflow_model_entity = model.load_from_mlflow()
        assert mlflow_model_entity.name == model.mlflow_name
        assert mlflow_model_entity.latest_versions
        assert len(mlflow_model_entity.latest_versions) == 1


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
