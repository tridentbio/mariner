"""
Dictionary like helper classes used my model_builder.storage
"""
from typing import Iterable, List, Mapping


class MappingView(object):
    """
    Dictionary like class
    """

    def __init__(self, mapping: Mapping, *args: List[str]) -> None:
        self._mapping = mapping
        self._args = args

    def _keys(self) -> Iterable:
        if len(self._args) == 0:
            return self._mapping.keys()
        return [arg for arg in self._args if arg in self._mapping]

    def __len__(self) -> int:
        return len(self._keys())

    def __repr__(self) -> str:
        mapping = {key: self._mapping[key] for key in self._keys()}
        return f"{self.__class__.__name__}({mapping})"

    __class_getitem__ = classmethod(type([]))


class KeysView(MappingView):
    """
    Extends MappingView with key iteration
    """

    def __iter__(self) -> Iterable:
        yield from self._keys()


class ValuesView(MappingView):
    """
    Extends MappingView with value iteration
    """

    def __iter__(self) -> Iterable:
        for key in self._keys():
            yield self._mapping[key]


class ItemsView(MappingView):
    """
    Extends MappingView with (key, value) iteration
    """

    def __iter__(self) -> Iterable:
        for key in self._keys():
            yield (key, self._mapping[key])
