import weakref
from collections.abc import MutableMapping
from copy import copy
from typing import Any, Callable, Dict, Iterable, List, Optional

from model_builder.utils import recursive_apply
from model_builder.views import ItemsView, KeysView, ValuesView


class BaseStorage(MutableMapping):
    def __init__(self, _mapping: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        super().__init__()
        self._mapping = {}

        # Setup all attributes that comes from _mapping
        for key, value in (_mapping or {}).items():
            setattr(self, key, value)

        # Transform all arguments passed by kwargs
        # in new atttributes for the base storage instance
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def _key(self) -> Any:
        return None

    def __len__(self) -> int:
        return len(self._mapping)

    def __getattr__(self, key: str) -> Any:
        if key == "_mapping":
            self._mapping = {}
            return self._mapping

        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "_parent":
            self.__dict__[key] = weakref.ref(value)
        elif key[:1] == "_":
            self.__dict__[key] = value
        else:
            self[key] = value

    def __delattr__(self, key: str) -> None:
        if key[:1] == "_":
            del self.__dict__[key]
        else:
            del self[key]

    def __getitem__(self, key: str) -> Any:
        return self._mapping[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if value is None and key in self._mapping:
            del self._mapping[key]
        elif value is not None:
            self._mapping[key] = value

    def __delitem__(self, key: str) -> None:
        if key in self._mapping:
            del self._mapping[key]

    def __iter__(self) -> Iterable:
        return iter(self._mapping)

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)

        for key, value in self.__dict__.items():
            out.__dict__[key] = value

        out._mapping = copy.copy(out._mapping)
        return out

    def __deepcopy__(self, memo):
        out = self.__class__.__new__(self.__class__)

        for key, value in self.__dict__.items():
            out.__dict__[key] = value

        out._mapping = copy.deepcopy(out._mapping, memo)
        return out

    def __getstate__(self) -> Dict[str, Any]:
        out = self.__dict__.copy()

        _parent = out.get("_parent", None)
        if _parent is not None:
            out["_parent"] = _parent()

        return out

    def __setstate__(self, mapping: Dict[str, Any]) -> None:
        for key, value in mapping.items():
            self.__dict__[key] = value

        _parent = self.__dict__.get("_parent", None)
        if _parent is not None:
            self.__dict__["_parent"] = weakref.ref(_parent)

    def __repr__(self) -> str:
        return repr(self._mapping)

    def keys(self, *args: List[str]) -> KeysView:
        return KeysView(self._mapping, *args)

    def values(self, *args: List[str]) -> ValuesView:
        return ValuesView(self._mapping, *args)

    def items(self, *args: List[str]) -> ItemsView:
        return ItemsView(self._mapping, *args)

    def apply_(self, function: Callable, *args: List[str]):
        for key, value in self.items(*args):
            self[key] = recursive_apply(value, function)
        return self

    def apply(self, function: Callable, *args: List[str]):
        for key, value in self.items(*args):
            self[key] = recursive_apply(value, function)
        return self
