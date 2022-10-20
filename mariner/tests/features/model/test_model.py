from app.features.model.schema.model import Model


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
