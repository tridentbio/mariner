import traceback
from typing import Any, List, Optional, Union, get_type_hints

import mlflow
import mlflow.exceptions
import pandas as pd
import pytorch_lightning as pl
import torch
from sqlalchemy.orm.session import Session
from app.logger import logger

from app.builder.dataset import CustomDataset
from app.builder.model import CustomModel
from app.core import mlflowapi
from app.features.dataset.crud import repo as dataset_repo
from app.features.dataset.exceptions import DatasetNotFound
from app.features.model import generate
from app.features.model.crud import repo
from app.features.model.deployments.crud import repo as deployment_repo
from app.features.model.deployments.schema import (
    Deployment,
    DeploymentCreate,
    DeploymentCreateRepo,
)
from app.features.model.exceptions import (
    ModelNameAlreadyUsed,
    ModelNotFound,
    ModelVersionNotFound,
)
from app.features.model.schema import layers_schema
from app.features.model.schema.configs import (
    ComponentAnnotation,
    ComponentOption,
    ModelSchema,
    ModelOptions,
)
from app.features.model.schema.model import (
    Model,
    ModelCreate,
    ModelCreateRepo,
    ModelFeaturesAndTarget,
    ModelsQuery,
    ModelVersion,
    ModelVersionCreateRepo,
)
from app.features.model.utils import get_class_from_path_string
from app.features.user.model import User as UserEntity
from app.schemas.api import ApiBaseModel
from torch_geometric.loader import DataLoader


def get_model_and_dataloader(
    db: Session, config: ModelSchema
) -> tuple[pl.LightningModule, DataLoader]:
    dataset = dataset_repo.get_by_name(db, config.dataset.name)
    assert dataset
    df = dataset.get_dataframe()
    dataset = CustomDataset(
        data=df,
        feature_columns=config.dataset.feature_columns,
        featurizers_config=config.featurizers,
    )
    dataloader = DataLoader(dataset, batch_size=len(df))
    model = CustomModel(config)
    return model, dataloader


class ForwardCheck(ApiBaseModel):
    stack_trace: Optional[str] = None
    output: Optional[Any] = None


def naive_check_forward_exception(db: Session, config: ModelSchema) -> ForwardCheck:
    """Given a model config, run it through a small set of the dataset
    and return the exception if any. It does not mean the training will
    run smoothly for all examples of the dataset, but it gives some
    confidence about the model trainability"""
    try:
        model, dataloader = get_model_and_dataloader(db, config)
        sample = next(iter(dataloader))
        output = model(sample)
        return ForwardCheck(output=output)
    except:  # noqa E722
        lines = traceback.format_exc(limit=5)
        return ForwardCheck(stack_trace=lines)


def create_model(
    db: Session,
    user: UserEntity,
    model_create: ModelCreate,
) -> Model:
    r"""
    Creates a model and a model version if passed with a
    non-existing `model_create.name`, otherwise creates a
    new model version.
    Triggers the creation of a RegisteredModel and a ModelVersion
    in MLFlow API's
    """
    client = mlflowapi.create_tracking_client()
    dataset = dataset_repo.get_by_name(db, model_create.config.dataset.name)
    if not dataset:
        raise DatasetNotFound()
    torchmodel = CustomModel(model_create.config)
    existingmodel = repo.get_by_name(db, model_create.name)
    mlflow_name = f"{user.id}-{model_create.name}"
    if existingmodel and existingmodel.created_by_id != user.id:
        raise ModelNameAlreadyUsed()

    if not existingmodel:
        regmodel, version = mlflowapi.create_registered_model(
            client,
            mlflow_name,
            torchmodel,
            description=model_create.model_description,
            version_description=model_create.model_version_description,
        )
    else:
        regmodel = mlflowapi.get_registry_model(mlflow_name, client=client)
        version = mlflowapi.create_model_version(
            client,
            existingmodel.mlflow_name,
            torchmodel,
            desc=model_create.model_version_description,
        )

    if not existingmodel:
        existingmodel = repo.create(
            db,
            obj_in=ModelCreateRepo(
                name=model_create.name,
                mlflow_name=mlflow_name,
                created_by_id=user.id,
                dataset_id=dataset.id,
                columns=[
                    ModelFeaturesAndTarget.construct(
                        column_name=feature_col.name,
                        column_type="feature",
                    )
                    for feature_col in model_create.config.dataset.feature_columns
                ]
                + [
                    ModelFeaturesAndTarget.construct(
                        column_name=model_create.config.dataset.target_column.name,
                        column_type="target",
                    )
                ],
            ),
        )

    if version:
        version = ModelVersion.from_orm(
            repo.create_model_version(
                db,
                version_create=ModelVersionCreateRepo(
                    mlflow_version=str(version.version),
                    model_id=existingmodel.id,
                    name=model_create.config.name,
                    mlflow_model_name=regmodel.name,
                    config=model_create.config,
                ),
            )
        )
    model = Model.from_orm(repo.get_by_name(db, existingmodel.name))
    model.description = regmodel.description
    return model


def create_model_deployment(
    db: Session, deployment_create: DeploymentCreate, user: UserEntity
) -> Deployment:
    """Create a deployment object that represents an endpoint responsible
    for model inferences

    TODO: Create authentication configuration for deployment
    """
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
    """Get all registered models"""
    models, total = repo.get_paginated(
        db,
        created_by_id=current_user.id,
        per_page=query.per_page,
        page=query.page,
        q=query.q,
    )
    models = [Model.from_orm(model) for model in models]
    for model in models:
        try:
            model.load_from_mlflow()
        except mlflow.exceptions.MlflowException as exp:
            logger.error("Error while getting model's mlflow data")
            logger.error(exp)

    return models, total


def get_documentation_link(class_path: str) -> Optional[str]:
    """Create documentation link for the class_paths of pytorch
    and pygnn objects"""

    def is_from_pygnn(class_path: str) -> bool:
        return class_path.startswith("torch_geometric.")

    def is_from_pytorch(class_path: str) -> bool:
        return class_path.startswith("torch.")

    if is_from_pygnn(class_path):
        return f"https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#{class_path}"  # noqa: E501
    elif is_from_pytorch(class_path):
        return f"https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#{class_path}"  # noqa: E501
    return None


def remove_section_by_identation(text: str, section_title: str) -> str:
    def count_tabs(line: str) -> int:
        total = 0
        for c in line:
            if c == " ":
                total += 1
            elif c == "\t":
                total += 2
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


def get_annotations_from_cls(
    cls_path: str, type=None, component=None
) -> ComponentOption:
    """Gives metadata information of the component implemented by `cls_path`
    Current metadata includes:
        - Docs and Docs' link
        - Number of inputs and outputs the component expects
        - Known rules for the component
    """
    docs_link = get_documentation_link(cls_path)
    cls = get_class_from_path_string(cls_path)
    docs = remove_section_by_identation(cls.__doc__, "Examples:")
    type_hints = {key: str(value) for key, value in get_type_hints(cls).items()}
    output_type_hint = str(type_hints.pop("return", None))
    return ComponentOption.construct(
        docs_link=docs_link,
        docs=docs,
        positional_inputs=type_hints,
        output_type=output_type_hint,
        class_path=cls_path,
        type=type,
        component=component,
    )


def get_model_options() -> ModelOptions:
    """Gets all component (featurizers and layer) options supported by the system,
    along with metadata about each"""
    layer_types = [layer.name for layer in generate.layers]
    featurizer_types = [f.name for f in generate.featurizers]

    def get_summary(
        cls_path: str,
    ) -> Union[layers_schema.LayersArgsType, layers_schema.FeaturizersArgsType, None]:
        for l in dir(layers_schema):
            class_def = getattr(layers_schema, l)
            if l.endswith("Summary") and class_def.type == cls_path:
                return class_def

    component_annotations: ModelOptions = []
    for type, class_path in zip(
        ["layer"] * len(layer_types) + ["featurizer"] * len(featurizer_types),
        layer_types + featurizer_types,
    ):
        summary = get_summary(class_path)
        assert summary, f"class Summary of {class_path} should not be None"
        component_annotations.append(
            get_annotations_from_cls(class_path, type=type, component=summary)
        )
    return component_annotations


class PredictRequest(ApiBaseModel):
    user_id: int
    model_version_id: int
    model_input: Any


def get_model_prediction(db: Session, request: PredictRequest) -> torch.Tensor:
    """(Slowly) Loads a model version and apply it to a sample input"""
    modelversion = ModelVersion.from_orm(
        repo.get_model_version(db, request.model_version_id)
    )
    if not modelversion:
        raise ModelVersionNotFound()
    df = pd.DataFrame.from_dict(request.model_input)
    dataset = CustomDataset(
        data=df,
        feature_columns=modelversion.config.dataset.feature_columns,
        featurizers_config=modelversion.config.featurizers,
    )
    dataloader = DataLoader(dataset, batch_size=len(df))
    modelinput = next(iter(dataloader))
    pyfuncmodel = mlflowapi.get_model_by_uri(modelversion.get_mlflow_uri())
    return pyfuncmodel(modelinput)


def get_model_version(db: Session, user: UserEntity, model_id: int) -> Model:
    """Gets a single model version"""
    modeldb = repo.get(db, model_id)
    if not modeldb or modeldb.created_by_id != user.id:
        raise ModelNotFound()
    model = Model.from_orm(modeldb)
    model.load_from_mlflow()
    return model


def delete_model(db: Session, user: UserEntity, model_id: int) -> Model:
    """Deletes a model and all it's artifacts"""
    model = repo.get(db, model_id)
    if not model or model.created_by_id != user.id:
        raise ModelNotFound()
    model = Model.from_orm(model)
    client = mlflow.tracking.MlflowClient()
    client.delete_registered_model(model.mlflow_name)
    repo.delete_by_id(db, model_id)
    return model
