# Temporary file to hold all extracted mariner schemas
from typing import Any, Dict, List, Literal, Union

import networkx as nx
import yaml
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    root_validator,
    validator,
)
from typing_extensions import Annotated

from model_builder import generate
from model_builder.components_query import (
    get_component_constructor_args_by_type,
)
from model_builder.layers_schema import FeaturizersType, LayersType
from model_builder.utils import CamelCaseModel, get_references_dict


class NumericalDataType(CamelCaseModel):
    domain_kind: Literal["numeric"] = Field("numeric")

    @validator("domain_kind")
    def check_domain_kind(cls, v):
        return "numeric"


class QuantityDataType(NumericalDataType):
    unit: str


class StringDataType(CamelCaseModel):
    domain_kind: Literal["string"] = Field("string")

    @validator("domain_kind")
    def check_domain_kind(cls, v):
        return "string"


class CategoricalDataType(CamelCaseModel):
    domain_kind: Literal["categorical"] = Field("categorical")
    classes: dict[Union[str, int], int]

    @validator("domain_kind")
    def check_domain_kind(cls, v):
        return "categorical"


class SmileDataType(CamelCaseModel):
    domain_kind: Literal["smiles"] = Field("smiles")

    @validator("domain_kind")
    def check_domain_kind(cls, v):
        return "smiles"


class DNADataType(CamelCaseModel):
    domain_kind: Literal["dna"] = Field("dna")

    @validator("domain_kind")
    def check_domain_kind(cls, v):
        return "dna"


class RNADataType(CamelCaseModel):
    domain_kind: Literal["rna"] = Field("rna")

    @validator("domain_kind")
    def check_domain_kind(cls, v):
        return "rna"


class ProteinDataType(CamelCaseModel):
    domain_kind: Literal["protein"] = Field("protein")

    @validator("domain_kind")
    def check_domain_kind(cls, v):
        return "protein"


# TODO: make data_type optional
class ColumnConfig(CamelCaseModel):
    name: str
    data_type: Union[
        QuantityDataType,
        NumericalDataType,
        StringDataType,
        SmileDataType,
        CategoricalDataType,
        DNADataType,
        RNADataType,
        ProteinDataType,
    ] = Field(...)


# TODO: move featurizers to feature_columns
class DatasetConfig(CamelCaseModel):
    name: str
    target_column: ColumnConfig
    feature_columns: List[ColumnConfig]


class UnknownComponentType(ValueError):
    """
    Raised when an unknown component type is detected

    Attributes:
        component_name: The id of the component with bad type
    """

    component_name: str

    def __init__(self, *args, component_name: str):
        super().__init__(*args)
        self.component_name = component_name


class MissingComponentArgs(ValueError):
    """
    Raised when there are missing arguments for a component.

    It's used by the frontend editor to provide accurate user feedback
    on what layer/featurizer went wrong (using the layer/featurizer id instead of json
    location)

    Attributes:
        component_name: component id that failed
        missing: list of fields that are missing
    """

    component_name: str
    missing: List[Union[str, int]]

    def __init__(self, *args, missing: List[Union[str, int]], component_name: str):
        super().__init__(*args)
        self.component_name = component_name
        self.missing = missing


AnnotatedLayersType = Annotated[LayersType, Field(discriminator="type")]
LossType = Literal["torch.nn.MSELoss", "torch.nn.CrossEntropyLoss"]

ALLOWED_CLASSIFIYNG_LOSSES = ["torch.nn.CrossEntropyLoss"]
ALLOWED_REGRESSOR_LOSSES = ["torch.nn.MSELoss"]


def is_regression(target_column: ColumnConfig):
    return target_column.data_type.domain_kind == "numeric"


def is_classifier(target_column: ColumnConfig):
    return target_column.data_type.domain_kind == "categorical"


class ModelSchema(CamelCaseModel):
    """
    A serializable neural net architecture
    """

    def is_regressor(self) -> bool:
        return is_regression(self.dataset.target_column)

    def is_classifier(self) -> bool:
        return is_classifier(self.dataset.target_column)

    @root_validator(pre=True)
    def check_types_defined(cls, values):
        layers = values.get("layers")
        featurizers = values.get("featurizers")
        layer_types = [layer.name for layer in generate.layers]
        for layer in layers:
            if not isinstance(layer, dict):
                layer = layer.dict()
            if layer["type"] not in layer_types:
                raise UnknownComponentType(
                    "A layer has unknown type", component_name=layer["name"]
                )
        if featurizers:
            featurizer_types = [featurizer.name for featurizer in generate.featurizers]
            for featurizer in featurizers:
                if not isinstance(featurizer, dict):
                    featurizer = featurizer.dict()
                if featurizer["type"] not in featurizer_types:
                    raise UnknownComponentType(
                        f"A featurizer has unknown type: {featurizer['type']}",
                        component_name=featurizer["name"],
                    )
        return values

    @root_validator(pre=True)
    def check_no_missing_args(cls, values):
        layers = values.get("layers")
        featurizers = values.get("featurizers")
        errors = []
        for layer in layers:
            if not isinstance(layer, dict):
                layer = layer.dict()
            args_cls = get_component_constructor_args_by_type(layer["type"])
            if not args_cls or "constructorArgs" not in layer:
                continue
            try:
                args_cls.validate(layer["constructorArgs"])
            except ValidationError as exp:
                errors += [
                    MissingComponentArgs(
                        missing=[missing_arg_name for missing_arg_name in error["loc"]],
                        component_name=layer["name"],
                    )
                    for error in exp.errors()
                    if error["type"] == "value_error.missing"
                ]
        if featurizers:
            for featurizer in featurizers:
                if not isinstance(featurizer, dict):
                    featurizer = featurizer.dict()
                args_cls = get_component_constructor_args_by_type(featurizer["type"])
                if not args_cls or "constructorArgs" not in featurizer:
                    continue
                try:
                    args_cls.validate(featurizer["constructorArgs"])
                except ValidationError as exp:
                    errors += [
                        MissingComponentArgs(
                            missing=[
                                missing_arg_name for missing_arg_name in error["loc"]
                            ],
                            component_name=featurizer["name"],
                        )
                        for error in exp.errors()
                        if error["type"] == "value_error.missing"
                    ]

        if len(errors) > 0:
            # TODO: raise all errors grouped in a single validation error
            raise errors[0]
        return values

    @validator("loss_fn", pre=True, always=True)
    def autofill_loss_gn(cls, value: str, values: Dict[str, Any]) -> Any:
        target_column = values["dataset"].target_column
        if is_classifier(target_column):
            if not value:
                return "torch.nn.CrossEntropyLoss"
            else:
                if value not in ALLOWED_CLASSIFIYNG_LOSSES:
                    raise ValidationError(
                        "loss is not valid for classifiers", model=ModelSchema
                    )
                return value
        elif is_regression(target_column):
            if not value:
                return "torch.nn.MSELoss"
            else:
                if value not in ALLOWED_REGRESSOR_LOSSES:
                    raise ValidationError(
                        "loss is not valid for regressors", model=ModelSchema
                    )
                return value
        raise ValueError(
            f"Can't determine task for output {target_column}."
            "Accepted output types are numeric or categorical."
        )

    name: str
    dataset: DatasetConfig
    layers: List[AnnotatedLayersType] = []
    featurizers: List[FeaturizersType] = []
    loss_fn: LossType = None

    def make_graph(self):
        g = nx.DiGraph()
        for feat in self.featurizers:
            g.add_node(feat.name)
        for layer in self.layers:
            g.add_node(layer.name)

        def _to_dict(arr: List[BaseModel]):
            return [item.dict() for item in arr]

        layers_and_featurizers = _to_dict(self.layers)
        if self.featurizers:
            layers_and_featurizers += _to_dict(self.featurizers)
        for feat in layers_and_featurizers:
            if "forward_args" not in feat:
                continue
            references = get_references_dict(feat["forward_args"])
            for key, value in references.items():
                if not value:  # optional forward_arg that was not set
                    continue
                if isinstance(value, str):
                    reference = value.split(".")[0]
                    g.add_edge(reference, feat["name"], attr=key)
                elif isinstance(value, list):
                    for reference in value:
                        reference = reference.split(".")[0]
                        g.add_edge(reference, feat["name"], attr=key)

                else:
                    continue
        return g

    @classmethod
    def from_yaml(cls, yamlstr):
        config_dict = yaml.safe_load(yamlstr)
        return ModelSchema(**config_dict)


if __name__ == "__main__":
    print(ModelSchema.schema_json())
