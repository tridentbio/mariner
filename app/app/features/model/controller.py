from typing import Any, List, Optional, Union, get_type_hints

import numpy as np
import pandas as pd
import torch
import torch_geometric
from sqlalchemy.orm.session import Session

import app.features.model.layers as mariner_layers
from app.core import mlflowapi
from app.features.model import generate
from app.features.model.builder import CustomModel
from app.features.model.crud import repo
from app.features.model.deployments.crud import repo as deployment_repo
from app.features.model.deployments.schema import (
    Deployment,
    DeploymentCreate,
    DeploymentCreateRepo,
)
from app.features.model.exceptions import ModelNotFound
from app.features.model.schema import layers_schema
from app.features.model.schema.configs import (
    LayerAnnotation,
    LayerRule,
    ModelOptions,
)
from app.features.model.schema.model import (
    Model,
    ModelCreate,
    ModelCreateRepo,
    ModelsQuery,
)
from app.features.model.utils import get_class_from_path_string
from app.features.user.model import User as UserEntity
from app.schemas.api import ApiBaseModel


def create_model(
    db: Session,
    user: UserEntity,
    model_create: ModelCreate,
) -> Model:
    client = mlflowapi.create_tracking_client()
    torchmodel = CustomModel(model_create.config)
    mlflow_reg_model, version = mlflowapi.create_registered_model(
        client,
        model_create.name,
        torchmodel,
        description=model_create.model_description,
        version_description=model_create.model_version_description,
    )
    model = Model.from_orm(
        repo.create(
            db,
            obj_in=ModelCreateRepo(name=model_create.name, created_by_id=user.id),
        )
    )
    assert version is not None
    model.set_from_mlflow_model(mlflow_reg_model, versions=[version] if version else [])
    return model


def create_model_deployment(
    db: Session, deployment_create: DeploymentCreate, user: UserEntity
) -> Deployment:
    model_name, latest_version = (
        deployment_create.model_name,
        deployment_create.model_version,
    )
    endpoint_name = f"{user.id}-{model_name}-{latest_version}"
    mlflowapi.create_deployment_with_endpoint(
        deployment_name=endpoint_name,
        model_uri=f"models:/{model_name}/{latest_version}",
    )
    deployment_entity = deployment_repo.create(
        db,
        obj_in=DeploymentCreateRepo(
            created_by_id=user.id,
            name=deployment_create.name,
            model_name=deployment_create.model_name,
            model_version=deployment_create.model_version,
        ),
    )
    return Deployment.from_orm(deployment_entity)


def get_models(db: Session, query: ModelsQuery, current_user: UserEntity):
    models, total = repo.get_paginated(
        db, created_by_id=current_user.id, per_page=query.per_page, page=query.page
    )
    models = [Model.from_orm(model) for model in models]
    for model in models:
        model.load_from_mlflow()
    return models, total


def get_documentation_link(class_path: str) -> Optional[str]:
    def is_from_pygnn(class_path: str) -> bool:
        return class_path.startswith("torch_geometric.")

    def is_from_pytorch(class_path: str) -> bool:
        return class_path.startswith("torch.")

    if is_from_pygnn(class_path):
        return f"https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#{class_path}"
    elif is_from_pytorch(class_path):
        return f"https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#{class_path}"
    return None


def remove_indentation_of_section(text: str, section_title: str) -> str:
    def count_tabs(line: str) -> int:
        # naive loop over chars
        total = 0
        for c in line:
            if c == " " or c == "\t":
                total += 1
            else:
                break
        return total

    lines = text.split("\n")
    start_idx = None
    tab_size = None
    end_idx = None
    for idx, line in enumerate(lines):
        if section_title in line:
            start_idx = idx
            end_idx = start_idx
            tab_size = count_tabs(line) + 4
            continue
        if tab_size is not None and count_tabs(line) >= tab_size:
            assert end_idx is not None
            end_idx = end_idx + 1
        elif start_idx is not None:
            break
    if start_idx is None or end_idx is None:
        return text
    return "\n".join(lines[:start_idx] + lines[end_idx + 1 :])


def get_inputs_outputs_and_rules(component_cls) -> tuple[int, int, List[LayerRule]]:
    rules = []
    if issubclass(component_cls, torch_geometric.nn.MessagePassing) or issubclass(
        component_cls, mariner_layers.GlobalPooling
    ):
        rules.append("graph-receiver")
        return 1, 1, rules
    elif issubclass(component_cls, mariner_layers.Concat):
        rules.append("inputs-same-type")
        return 2, 1, rules
    elif issubclass(component_cls, torch.nn.Module):
        type_hints_keys = get_type_hints(component_cls.forward).keys()
        inputs = len(type_hints_keys) - int("return" in type_hints_keys)
        return inputs, 1, rules
    else:
        type_hints_keys = get_type_hints(component_cls.__call__).keys()
        inputs = len(type_hints_keys) - int("return" in type_hints_keys)
        return inputs, 1, rules


def get_annotations_from_cls(cls_path: str) -> LayerAnnotation:
    docs_link = get_documentation_link(cls_path)
    cls = get_class_from_path_string(cls_path)
    docs = remove_indentation_of_section(cls.__doc__, "Examples:")
    inputs, outputs, rules = get_inputs_outputs_and_rules(cls)
    return LayerAnnotation(
        docs_link=docs_link,
        docs=docs,
        num_inputs=inputs,
        num_outputs=outputs,
        rules=rules,
        class_path=cls_path
    )


def get_model_options() -> ModelOptions:
    layers = []
    featurizers = []
    layer_types = [layer.name for layer in generate.layers]
    featurizer_types = [f.name for f in generate.featurizers]
    component_annotations = []
    for class_name in dir(layers_schema):
        if class_name.endswith("ArgsTemplate"):
            cls = getattr(layers_schema, class_name)
            instance = cls()
            if instance.type in layer_types:
                layers.append(cls())
            if instance.type in featurizer_types:
                featurizers.append(cls())
    for class_path in layer_types + featurizer_types:
        annotation = get_annotations_from_cls(class_path)
        component_annotations.append(annotation)
    return ModelOptions(
        layers=layers,
        featurizers=featurizers,
        component_annotations=component_annotations,
    )


ModelOutput = Union[np.ndarray, pd.Series, pd.DataFrame, List]


class PredictRequest(ApiBaseModel):
    user_id: int
    model_name: str
    version: Optional[Union[str, int]]
    model_input: Any


def get_model_prediction(db: Session, request: PredictRequest) -> ModelOutput:
    model = repo.get_by_name_from_user(db, request.user_id, request.model_name)
    if not model:
        raise ModelNotFound()
    model = Model.from_orm(model)
    pyfuncmodel = mlflowapi.get_model(model, request.version)
    # TODO: fix this cheat
    # refactor torch custom dataset to receive a dataframe
    # instead of a dataset. save the model config along with
    # the model in the database. This should be enough
    # to instantiate the torch dataset, then building a batch
    # as the model expects it
    from app.tests.conftest import mock_dataset_item

    mocked_input = mock_dataset_item()
    return pyfuncmodel(mocked_input)
