"""
Models service
"""

import traceback
from inspect import Parameter, signature
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    get_type_hints,
)
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
from mariner.exceptions.model_exceptions import (
    InvalidDataframe,
    ModelVersionNotTrained,
)
from mariner.schemas.api import ApiBaseModel
from mariner.schemas.model_schemas import (
    ComponentOption,
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
from mariner.validation.functions import is_valid_smiles_series
from model_builder import generate, layers_schema
from model_builder.dataset import CustomDataset
from model_builder.model import CustomModel
from model_builder.schemas import DatasetConfig, SmileDataType
from model_builder.utils import get_class_from_path_string


def get_model_and_dataloader(
    db: Session, config: ModelSchema
) -> tuple[pl.LightningModule, DataLoader]:
    """Gets the CustomModel and the DataLoader needed to train
    a model

    Args:
        db: connection to the database
        config: model config

    Returns:
        tuple[pl.LightningModule, DataLoader]
    """
    dataset_entity = dataset_store.get_by_name(db, config.dataset.name)
    assert dataset_entity, f"No dataset found with name {config.dataset.name}"
    df = dataset_entity.get_dataframe()
    dataset = CustomDataset(data=df, config=config)
    dataloader = DataLoader(dataset, batch_size=1)
    model = CustomModel(config)
    return model, dataloader


class ForwardCheck(ApiBaseModel):
    """Response to a request to check if a model version forward works"""

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
    except:  # noqa E722 pylint: disable=W0702
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
    dataset = dataset_store.get_by_name(db, model_create.config.dataset.name)
    if not dataset:
        raise DatasetNotFound()

    client = mlflowapi.create_tracking_client()
    # Handle case where model_create.name refers to existing model
    existingmodel = model_store.get_by_name_from_user(db, model_create.name, user_id=user.id)
    if existingmodel:
        model_store.create_model_version(
            db,
            ModelVersionCreateRepo(
                mlflow_version=None,
                mlflow_model_name=existingmodel.mlflow_name,
                model_id=existingmodel.id,
                name=model_create.config.name,
                config=model_create.config,
                description=model_create.model_version_description,
            ),
        )
        return Model.from_orm(existingmodel)

    # Handle cases of creating a new model in the registry
    mlflow_name = f"{user.id}-{model_create.name}-{uuid4()}"

    try:
        mlflowapi.create_registered_model(
            client,
            name=mlflow_name,
            description=model_create.model_description,
        )
    except mlflow.exceptions.MlflowException as exp:
        if exp.error_code == mlflow.exceptions.RESOURCE_ALREADY_EXISTS:
            raise ModelNameAlreadyUsed(
                "A model with that name is already in use by mlflow"
            ) from exp
        raise

    model = model_store.create(
        db,
        obj_in=ModelCreateRepo(
            name=model_create.name,
            description=model_create.model_description,
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
                    column_name=target_column.name,
                    column_type="target",
                )
                for target_column in model_create.config.dataset.target_columns
            ],
        ),
    )
    model_store.create_model_version(
        db,
        ModelVersionCreateRepo(
            mlflow_version=None,
            mlflow_model_name=mlflow_name,
            model_id=model.id,
            name=model_create.config.name,
            config=model_create.config,
            description=model_create.model_version_description,
        ),
    )

    model = Model.from_orm(model)
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
    parsed_models = [Model.from_orm(model) for model in models]
    return parsed_models, total


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
        type=None,  # type: ignore
        component=None,  # type: ignore
        default_args=None,  # type: ignore
    )


def get_model_options() -> ModelOptions:
    """Gets all component (featurizers and layer) options supported by the system,
    along with metadata about each"""
    layer_types = [layer.name for layer in generate.layers]
    featurizer_types = [f.name for f in generate.featurizers]

    def get_default_values(summary_name: str) -> Dict[str, Any]:
        try:
            class_args = getattr(
                layers_schema, summary_name.replace("Summary", "ConstructorArgs")
            )
            args = {
                arg.name: arg.default
                for arg in signature(class_args).parameters.values()
                if arg.default != Parameter.empty
            }
            return args
        except AttributeError:
            return None

    def get_summary(
        cls_path: str,
    ) -> Union[None, layers_schema.LayersArgsType, layers_schema.FeaturizersArgsType]:
        for schema_exported in dir(layers_schema):
            if (
                schema_exported.endswith("Summary")
                and not schema_exported.endswith("ForwardArgsSummary")
                and not schema_exported.endswith("ConstructorArgsSummary")
            ):
                class_def = getattr(layers_schema, schema_exported)
                default_args = get_default_values(schema_exported)
                instance = class_def()
                if instance.type and instance.type == cls_path:
                    return instance, default_args
        raise RuntimeError(f"Schema for {cls_path} not found")

    component_annotations: List[ComponentOption] = []

    def make_component(class_path: str, type_: Literal["layer", "featurizer"]):
        summary, default_args = get_summary(class_path)
        assert summary, f"class Summary of {class_path} should not be None"
        option = get_annotations_from_cls(class_path)
        option.type = type_
        option.component = summary
        option.default_args = default_args
        return option

    for class_path in layer_types:
        component_annotations.append(make_component(class_path, "layer"))

    for class_path in featurizer_types:
        component_annotations.append(make_component(class_path, "featurizer"))

    return component_annotations


class PredictRequest(ApiBaseModel):
    """Payload of a prediction request"""

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
    column_configs = dataset_config.feature_columns + dataset_config.target_columns
    result: List[Tuple[str, InvalidReason]] = []
    for column_config in column_configs:
        data_type = column_config.data_type
        if isinstance(data_type, SmileDataType):
            series = df[column_config.name]
            if not is_valid_smiles_series(series):
                result.append((column_config.name, "invalid-smiles-series"))
    return result


def get_model_prediction(
    db: Session, request: PredictRequest
) -> Dict[str, torch.Tensor]:
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
    if not modelversion.mlflow_version:
        raise ModelVersionNotTrained()
    df = pd.DataFrame.from_dict(request.model_input, dtype=float)
    broken_checks = _check_dataframe_conforms_dataset(df, modelversion.config.dataset)
    if len(broken_checks) > 0:
        raise InvalidDataframe(
            f"dataframe failed {len(broken_checks)} checks",
            reasons=[f"{col_name}: {rule}" for col_name, rule in broken_checks],
        )
    dataset = CustomDataset(data=df, config=modelversion.config, target=False)
    dataloader = DataLoader(dataset, batch_size=len(df))
    modelinput = next(iter(dataloader))
    pyfuncmodel = mlflowapi.get_model_by_uri(modelversion.get_mlflow_uri())
    return pyfuncmodel.predict_step(modelinput)


def get_model(db: Session, user: UserEntity, model_id: int) -> Model:
    """Gets a single model"""
    modeldb = model_store.get(db, model_id)
    if not modeldb or modeldb.created_by_id != user.id:
        raise ModelNotFound()
    model = Model.from_orm(modeldb)
    return model


def delete_model(db: Session, user: UserEntity, model_id: int) -> Model:
    """Deletes a model and all it's artifacts"""
    model = model_store.get(db, model_id)
    if not model or model.created_by_id != user.id:
        raise ModelNotFound()
    client = mlflow.tracking.MlflowClient()
    client.delete_registered_model(model.mlflow_name)
    parsed_model = Model.from_orm(model)
    model_store.delete_by_id(db, model_id)
    return parsed_model
