from pathlib import Path

from fleet.torch_ import TorchFunctions, TorchModelSpec, TorchTrainingConfig

root = Path(".") / "tests" / "data" / "yml"
torch_model_specs = [
    root / "small_regressor_schema.yaml",
    # root / "binary_classification_model.yaml",
    # root / "binary_classification_model.yaml",
    # root / "categorical_features_model.yaml",
    # root / "dna_example.yml",
    # root / "model_fails_on_training.yml",
    # root / "modelv2.yaml",
    # root / "multiclass_classification_model.yaml",
    # root / "multitarget_classification_model.yaml",
    # root / "small_classifier_schema.yaml",
    # root / "test_model_with_from_smiles.yaml",
]


class TestTorchFunctions:

    torch_functions = TorchFunctions()

    def test_train(
        self, model_spec: TorchModelSpec, training_params: TorchTrainingConfig
    ):
        self.torch_functions.train(spec=model_spec, params=training_params)
