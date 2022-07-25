from typing import Any, Dict


def split_module_export(classpath: str) -> tuple[str, str]:
    words = classpath.split(".")
    module = ".".join(words[:-1])
    export = words[-1]
    return module, export


def get_class_from_path_string(pathstring: str) -> Any:
    module_name, export = split_module_export(pathstring)
    code = f"""
from {module_name} import {export}
references['cls'] = {export}
"""
    references: Dict[str, Any] = {}
    exec(code, globals(), {"references": references})
    return references["cls"]
