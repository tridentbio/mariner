"""
Provides base classes to help with YAML convertible pydantic schemas.
"""
from pathlib import Path
from typing import Union

import yaml
from pydantic import ValidationError


class YAML_Model:  # pylint: disable=invalid-name
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
            try:
                return cls.from_yaml_str(yaml_str)
            except ValidationError as e:
                raise AttributeError(f"Error parsing {yamlpath}") from e
