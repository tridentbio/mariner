import traceback
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, get_type_hints
from uuid import uuid4

import mlflow
import mlflow.exceptions
import pandas as pd
import pytorch_lightning as pl
import torch
from sqlalchemy.orm.session import Session
from torch_geometric.loader import DataLoader

from mariner.core import mlflowapi
from mariner.entities.user import User as UserEntity
from mariner.exceptions import (
    DatasetNotFound,
    ModelNameAlreadyUsed,
    ModelNotFound,
    ModelVersionNotFound,
)
from mariner.exceptions.model_exceptions import InvalidDataframe
from mariner.logger import logger
from mariner.schemas.api import ApiBaseModel
from mariner.schemas.model_schemas import (
    ComponentOption,
    LossOption,
    Model,
    ModelCreate,
    ModelCreateRepo,
    ModelFeaturesAndTarget,
    ModelOptions,
    ModelSchema,
    ModelsQuery,
    ModelVersion,
    ModelVersionCreateRepo,
)
from mariner.stores.dataset_sql import dataset_store
from mariner.stores.model_sql import model_store
from mariner.validation import is_valid_smiles_series
from model_builder import generate, layers_schema
from model_builder.dataset import CustomDataset
from model_builder.model import CustomModel
from model_builder.schemas import DatasetConfig, SmileDataType
from model_builder.utils import get_class_from_path_string


def get_model_and_dataloader(
    db: Session, config: ModelSchema
) -> tuple[pl.LightningModule, DataLoader]:
    dataset = dataset_store.get_by_name(db, config.dataset.name)
    assert dataset, f"No dataset found with name {config.dataset.name}"
    df = dataset.get_dataframe()
    dataset = CustomDataset(
        data=df,
        feature_columns=config.dataset.feature_columns,
        featurizers_config=config.featurizers,
    )
    dataloader = DataLoader(dataset, batch_size=1)
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
        lines = traceback.format_exc()
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
    dataset = dataset_store.get_by_name(db, model_create.config.dataset.name)
    if not dataset:
        raise DatasetNotFound()
    torchmodel = CustomModel(model_create.config)
    existingmodel = model_store.get_by_name(db, model_create.name)
    mlflow_name = f"{user.id}-{model_create.name}-{uuid4()}"
    if existingmodel and existingmodel.created_by_id != user.id:
        raise ModelNameAlreadyUsed()

    if not existingmodel:
        try:
            regmodel, version = mlflowapi.create_registered_model(
                client,
                mlflow_name,
                torchmodel,
                description=model_create.model_description,
                version_description=model_create.model_version_description,
            )
        except mlflow.exceptions.MlflowException as exp:
            if exp.error_code == mlflow.exceptions.RESOURCE_ALREADY_EXISTS:
                raise ModelNameAlreadyUsed(
                    "A model with that name is already in use by mlflow"
                )
            raise
    else:
        regmodel = mlflowapi.get_registry_model(
            existingmodel.mlflow_name, client=client
        )
        version = mlflowapi.create_model_version(
            client,
            existingmodel.mlflow_name,
            torchmodel,
            desc=model_create.model_version_description,
        )

    if not existingmodel:
        existingmodel = model_store.create(
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
            model_store.create_model_version(
                db,
                version_create=ModelVersionCreateRepo(
                    mlflow_version=str(version.version),
                    model_id=existingmodel.id,
                    name=model_create.config.name,
                    mlflow_model_name=regmodel.name,
                    config=model_create.config,
                    description=model_create.model_version_description,
                ),
            )
        )
    model = Model.from_orm(model_store.get_by_name(db, existingmodel.name))
    model.description = regmodel.description
    return model


def get_models(db: Session, query: ModelsQuery, current_user: UserEntity):
    """Get all registered models"""
    models, total = model_store.get_paginated(
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


def get_annotations_from_cls(cls_path: str) -> ComponentOption:
    """Gives metadata information of the component implemented by `cls_path`"""
    docs_link = get_documentation_link(cls_path)
    cls = get_class_from_path_string(cls_path)
    try:
        docs = generate.sphinxfy(cls_path)
    except generate.EmptySphinxException:
        docs = cls_path
    forward_type_hints = {}
    if "forward" in dir(cls):
        forward_type_hints = get_type_hints(getattr(cls, "forward"))
    elif "__call__" in dir(cls):
        forward_type_hints = get_type_hints(getattr(cls, "__call__"))
    output_type_hint = forward_type_hints.pop("return", None)
    return ComponentOption.construct(
        docs_link=docs_link,
        docs=docs,
        output_type=str(output_type_hint) if output_type_hint else None,
        class_path=cls_path,
    )


def get_model_options() -> ModelOptions:
    """Gets all component (featurizers and layer) options supported by the system,
    along with metadata about each"""
    layer_types = [layer.name for layer in generate.layers]
    featurizer_types = [f.name for f in generate.featurizers]

    def get_summary(
        cls_path: str,
    ) -> Union[layers_schema.LayersArgsType, layers_schema.FeaturizersArgsType, None]:
        for schema_exported in dir(layers_schema):
            if (
                schema_exported.endswith("Summary")
                and not schema_exported.endswith("ForwardArgsSummary")
                and not schema_exported.endswith("ConstructorArgsSummary")
            ):
                class_def = getattr(layers_schema, schema_exported)
                instance = class_def()
                if instance.type and instance.type == cls_path:
                    return instance

    component_annotations = []
    for type, class_path in zip(
        ["layer"] * len(layer_types) + ["featurizer"] * len(featurizer_types),
        layer_types + featurizer_types,
    ):
        summary = get_summary(class_path)
        assert summary, f"class Summary of {class_path} should not be None"
        option = get_annotations_from_cls(class_path)
        option.type = type
        option.component = summary
        component_annotations.append(option)
    return component_annotations


class PredictRequest(ApiBaseModel):
    user_id: int
    model_version_id: int
    model_input: Any


InvalidReason = Literal["invalid-smiles-series"]


def _check_dataframe_conforms_dataset(
    df: pd.DataFrame, dataset_config: DatasetConfig
) -> List[Tuple[str, InvalidReason]]:
    """Checks if dataframe conforms to data types specified in dataset_config

    Args:
        df: dataframe to be checked
        dataset_config: model builder dataset configuration used in model building
    """
    column_configs = dataset_config.feature_columns + [dataset_config.target_column]
    result = []
    for column_config in column_configs:
        data_type = column_config.data_type
        if isinstance(data_type, SmileDataType):
            series = df[column_config.name]
            if not is_valid_smiles_series(series):
                result.append((column_config.name, "invalid-smiles-series"))
    return result


def get_model_prediction(db: Session, request: PredictRequest) -> torch.Tensor:
    """(Slowly) Loads a model version and apply it to a sample input

    Args:
        db: Session with the database
        request: prediction request data

    Returns:
        torch.Tensor: model output

    Raises:
        ModelVersionNotFound: If the version from request.model_version_id
        is not in the database
        InvalidDataframe: If the dataframe fails one or more checks
    """
    modelversion = ModelVersion.from_orm(
        model_store.get_model_version(db, request.model_version_id)
    )
    if not modelversion:
        raise ModelVersionNotFound()
    df = pd.DataFrame.from_dict(request.model_input, dtype=float)
    broken_checks = _check_dataframe_conforms_dataset(df, modelversion.config.dataset)
    if len(broken_checks) > 0:
        raise InvalidDataframe(
            f"dataframe failed {len(broken_checks)} checks",
            reasons=[f"{col_name}: {rule}" for col_name, rule in broken_checks],
        )
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
    modeldb = model_store.get(db, model_id)
    if not modeldb or modeldb.created_by_id != user.id:
        raise ModelNotFound()
    model = Model.from_orm(modeldb)
    model.load_from_mlflow()
    return model


def delete_model(db: Session, user: UserEntity, model_id: int) -> Model:
    """Deletes a model and all it's artifacts"""
    model = model_store.get(db, model_id)
    if not model or model.created_by_id != user.id:
        raise ModelNotFound()
    model = Model.from_orm(model)
    client = mlflow.tracking.MlflowClient()
    client.delete_registered_model(model.mlflow_name)
    model_store.delete_by_id(db, model_id)
    return model
