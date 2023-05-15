"""
Provides base classes to help with YAML convertible pydantic schemas.
"""
from pathlib import Path
from typing import Union

import yaml


class YAML_Model:
    @classmethod
    def from_yaml_str(cls, yamlstr):
        """Parses a model schema object directly form a yaml str

        Args:
            yamlstr (str): yaml str
        """
        config_dict = yaml.safe_load(yamlstr)
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yamlpath: Union[str, Path]):
        """Parses a model schema object directly from a yaml file

        Args:
            yamlpath (str, Path): file containing model in yaml format
        """
        with open(yamlpath, "rU") as f:
            yaml_str = f.read()
            return cls.from_yaml_str(yaml_str)
