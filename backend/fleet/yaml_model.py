"""
Provides base classes to help with YAML convertible pydantic schemas.
"""
from pathlib import Path
from typing import Union

import yaml
from pydantic import BaseModel


class YAML_Model(BaseModel):  # pylint: disable=invalid-name
    """
    Adds methods to handle the model as YAML files.
    """

    @classmethod
    def from_yaml_str(cls, yamlstr):
        """Parses a model schema object directly form a yaml str

        Args:
            yamlstr (str): yaml str
        """
        config_dict = yaml.safe_load(yamlstr)
        return cls.parse_obj(config_dict)

    @classmethod
    def from_yaml(cls, yamlpath: Union[str, Path]):
        """Parses a model schema object directly from a yaml file

        Args:
            yamlpath (str, Path): file containing model in yaml format
        """
        with open(yamlpath, "rU", encoding="utf-8") as file:
            yaml_str = file.read()
            return cls.from_yaml_str(yaml_str)

    def to_yaml_str(self) -> str:
        """Converts the model schema object to a yaml string

        Returns:
            str: yaml string
        """
        return yaml.dump(self.dict())
