from tests.fixtures.model import model_config

DEFAULT_LOSS_MAP = {
    "regression": "torch.nn.MSELoss",
    "binary": "torch.nn.BCEWithLogitsLoss",
    "multiclass": "torch.nn.CrossEntropyLoss",
}


def test_schema_autofills_lossfn():
    regressor_schema = model_config(model_type="regressor")
    classifier_schema = model_config(model_type="classifier")
    target_columns = (
        regressor_schema.dataset.target_columns
        + classifier_schema.dataset.target_columns
    )
    for target_column in target_columns:
        assert (
            target_column.loss_fn
            == DEFAULT_LOSS_MAP[target_column.column_type]
        ), f"loss_fn for {target_column.name} was not set to the {target_column.column_type} default"
