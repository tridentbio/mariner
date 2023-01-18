"""Tests the training actor"""


class TestTrainingActor:
    def test_train(self):
        """Checks if training is performed succesfully by training
        actor on valid dataset and model configuration"""
        ...

    def test_persists_metrics(self):
        """Checks wheter metrics can be found in expected
        mlflow location (db) and mariner (db)"""
        ...

    def test_persists_model(self):
        """Checks wheter model can be correctly loaded from mlflow
        registry (by model and model version). Checks if logged models
        are in the expected s3 artifact path, and mariner model version
        entity is mapping to the trained mlflow model version"""
        ...
