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


def get_inputs_from_mask_ltr(arr, mask):
    """
    >>> get_inputs_from_mask_ltr(['a', 'b', 'c'], 0b011)
    >>> ['b', 'c']
    """
    bit_idx = 0b1
    result = []
    for idx, el in enumerate(arr[::-1]):
        position = 1 << idx
        if position & mask != 0:
            result.append(el)
        bit_idx = bit_idx << 1
    return result[::-1]
