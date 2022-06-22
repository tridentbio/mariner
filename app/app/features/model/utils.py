def split_module_export(classpath: str) -> str:
    words = classpath.split(".")
    module = ".".join(words[:-1])
    export = words[-1]
    return module, export


def get_class_from_path_string(pathstring: str):
    module_name, export = split_module_export(pathstring)
    code = f"""
from {module_name} import {export}
references['cls'] = {export}
"""
    references = {}
    exec(code, globals(), {"references": references})
    return references["cls"]
