def get_module_name(classpath: str) -> str:
    return ".".join(classpath.split(".")[:-1])


def get_class_from_path_string(pathstring: str):
    module_name = get_module_name(pathstring)
    code = f"""
import {module_name}
references['cls'] = {pathstring}
"""
    references = {}
    exec(code, globals(), {"references": references})
    return references["cls"]
