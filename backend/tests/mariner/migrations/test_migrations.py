from pydantic import ValidationError

from fleet.torch_.schemas import TorchModelSpec

payload = {
    "name": "afoyucpx",
    "dataset": {
        "name": "bcuberfh",
        "targetColumns": [
            {
                "name": "large_petal_length",
                "dataType": {"domainKind": "categorical", "classes": {"0": 0, "1": 1}},
                "outModule": "Linear-7",
                "lossFn": "torch.nn.BCEWithLogitsLoss",
                "columnType": "binary",
            },
            {
                "name": "species",
                "dataType": {
                    "domainKind": "categorical",
                    "classes": {"0": 0, "1": 1, "2": 2},
                },
                "outModule": "Linear-6",
                "lossFn": "torch.nn.CrossEntropyLoss",
                "columnType": "multiclass",
            },
            {
                "name": "petal_length",
                "dataType": {"domainKind": "numeric", "unit": "mole"},
                "outModule": "Linear-5",
                "lossFn": "torch.nn.MSELoss",
                "columnType": "regression",
            },
        ],
        "featureColumns": [
            {
                "name": "sepal_length",
                "dataType": {"domainKind": "numeric", "unit": "mole"},
            },
            {
                "name": "sepal_width",
                "dataType": {"domainKind": "numeric", "unit": "mole"},
            },
        ],
    },
    "layers": [
        {
            "type": "model_builder.layers.Concat",
            "name": "Concat-0",
            "constructorArgs": {"dim": 1},
            "forwardArgs": {"xs": ["$sepal_length", "$sepal_width"]},
        },
        {
            "type": "torch.nn.Linear",
            "name": "Linear-1",
            "constructorArgs": {"in_features": 2, "out_features": 16, "bias": True},
            "forwardArgs": {"input": "$Concat-0"},
        },
        {
            "type": "torch.nn.ReLU",
            "name": "ReLU-2",
            "constructorArgs": {"inplace": False},
            "forwardArgs": {"input": "$Linear-1"},
        },
        {
            "type": "torch.nn.Linear",
            "name": "Linear-3",
            "constructorArgs": {"in_features": 16, "out_features": 16, "bias": True},
            "forwardArgs": {"input": "$ReLU-2"},
        },
        {
            "type": "torch.nn.ReLU",
            "name": "ReLU-4",
            "constructorArgs": {"inplace": False},
            "forwardArgs": {"input": "$Linear-3"},
        },
        {
            "type": "torch.nn.Linear",
            "name": "Linear-5",
            "constructorArgs": {"in_features": 16, "out_features": 1, "bias": True},
            "forwardArgs": {"input": "$ReLU-4"},
        },
        {
            "type": "torch.nn.Linear",
            "name": "Linear-6",
            "constructorArgs": {"in_features": 16, "out_features": 3, "bias": True},
            "forwardArgs": {"input": "$ReLU-4"},
        },
        {
            "type": "torch.nn.Linear",
            "name": "Linear-7",
            "constructorArgs": {"in_features": 16, "out_features": 1, "bias": True},
            "forwardArgs": {"input": "$ReLU-4"},
        },
    ],
    "featurizers": [],
}


def convert_to_spec(old_model_schema: dict) -> TorchModelSpec:
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


def test_convert_to_spec():
    convert_to_spec(payload)
