from ast import literal_eval
from typing import List, Literal

import yaml

from app.features.model.utils import get_class_from_path_string
from app.features.model.layers_schema import LayersType, FeaturizersType
from app.schemas.api import ApiBaseModel



class Tuple(str):
    val: tuple

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(examples=["(1,)", "(1,1,0)"])

    @classmethod
    def validate(cls, v):
        try:
            t = literal_eval(v)
            if not isinstance(t, tuple):
                raise ValueError("Tuple(s), s should evaluate to a tuple")
            return cls(v)
        except Exception:
            raise


class CycleInGraphException(ApiBaseModel):
    """
    Raised when cycles are detected in the computational
    graph of the model
    """

    code = 200
    message = "There is a cycle in the graph"


class DatasetConfig(ApiBaseModel):
    name: str
    target_column: str
    feature_columns: List[str]



class ModelConfig(ApiBaseModel):
    name: str
    dataset: DatasetConfig
    featurizers: List[FeaturizersType]
    layers: List[LayersType]

    # TODO: validate if layer names are unique

    @classmethod
    def from_yaml(cls, yamlstr):
        config_dict = yaml.safe_load(yamlstr)
        return ModelConfig.parse_obj(config_dict)


if __name__ == "__main__":
    print(ModelConfig.schema_json())
