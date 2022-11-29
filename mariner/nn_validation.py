"""
Helps model building providing validation to needed
python/neural-networks queries, such as if a type
matches against some other type or 2 matrix shapes
are multipliable


Since this should be a lightweight module, it would
be best to have it separate apart later, in such a way
that is possible to load it only with it's dependencies
"""

import logging
import sys
import typing

import torch
from humps import camel
from pydantic import BaseModel, ValidationError
from pydantic.class_validators import root_validator
from pydantic.error_wrappers import ErrorWrapper


class CheckerBaseModel(BaseModel):
    class Config:
        alias_generator = camel.case
        allow_population_by_field_name = True
        allow_population_by_alias = True
        orm_mode = True
        underscore_attrs_are_private = True


def _get_type(type_str: str) -> typing.Type:
    type_str = type_str.replace("<class '", "")
    type_str = type_str.replace("'>", "")
    if type_str in ["None", "NoneType"]:
        type_str = "type(None)"
    elif type_str.endswith("?"):
        type_str = type_str[:-1]
        type_str = f"typing.Optional[{type_str}]"
    return eval(type_str, {"typing": typing, "torch": torch})


class Memo:
    globals: typing.Dict[str, typing.Any]
    locals: typing.Dict[str, typing.Any]

    def __init__(self, globals: typing.Any, locals: typing.Any):
        self.globals = globals
        self.locals = locals


def from_frame(depth: int = 0) -> Memo:
    frame = sys._getframe(depth)
    return Memo(globals=frame.f_globals, locals=frame.f_locals)


def get_origin_type_and_args(annotation: typing.Any, memo: Memo):
    """Get's the origin_type and args from a type object, respects typing.ForwardRef

    Inspired by https://github.com/agronholm/typeguard/blob/833c371792d2a5e65b5a48ce65b52c5916a02ab6/src/typeguard/_checkers.py#L528 noqa

    Args:
        annotation: type object
        memo: the globals and locals that should be used while
        evaluating code
    """
    if isinstance(annotation, typing.ForwardRef):
        annotation = annotation._evaluate(memo.globals, memo.locals, frozenset())
    origin_type = typing.get_origin(annotation)
    if origin_type is typing.Annotated:
        annotation, *extras = typing.get_args(annotation)
        origin_type = typing.get_origin(annotation)
    elif origin_type is not None:
        args = typing.get_args(annotation)
    else:
        origin_type = annotation
        args = ()
    return origin_type, args


LOG = logging.getLogger(__name__)


def check_union(
    origin_type: typing.Any,
    args: typing.Tuple[typing.Any, ...],
    union_args: typing.Tuple[typing.Any, ...],
) -> bool:
    """Type checks information with unions

    Checks origin_type and args gotten from a type object
    against and array of possible typings

    Args:
        origin_type: origin_type of what is being checked
        args: type args of what is being checked
        union_args: array of valid types to check against

    Returns:
        bool: true if the origin_type and args make a valid
        type in union of union_arsg
    """
    if type(None) in args and origin_type is None:
        return True
    return origin_type in union_args


def check_types(type_a: str, type_b: str, memo: Memo) -> bool:
    """Checks 2 python types agains each other

    Validate type_a and type_b as types or classes.
    <class 'int'> -> int
    <class 'int'>? -> Optional[int]

    Args:
        type_a: string of type to be checked
        type_b: string of type to be checked
        memo: object with globals and locals for evaluation

    Returns:
        bool: True if type_a and type_b match

    Raises:
        NotImplementedError: If the checked types fall into some space
        for which we can't tell they match
        Exception: assumptions over inputs are broken, e.g. client
        sends giberrish.
    """
    type_a = _get_type(type_a)
    type_b = _get_type(type_b)
    a_origin, a_args = get_origin_type_and_args(type_a, memo)
    b_origin, b_args = get_origin_type_and_args(type_b, memo)
    # LOG.setLevel(logging.DEBUG)
    LOG.debug("a (type): %s", type_a)
    LOG.debug("a_origin: %s", a_origin)
    LOG.debug("a_args:   %s", a_args)
    LOG.debug("b (type): %s", type_b)
    LOG.debug("b_origin: %s", b_origin)
    LOG.debug("b_args:   %s", b_args)

    if a_origin is None and b_origin is None:
        return type_a == type_b
    elif a_origin == b_origin and a_args == b_args:  # TODO: good default case
        return True
    elif a_origin is typing.Union and b_origin is typing.Union:
        return a_args == b_args
    elif a_origin == typing.Union and b_origin is not None:
        return check_union(origin_type=b_origin, args=b_args, union_args=a_args)
    elif a_origin == typing.Union and b_origin is None:
        return check_union(origin_type=type_b, args=(), union_args=a_args)
    elif b_origin == typing.Union and a_origin is not None:
        return check_union(origin_type=a_origin, args=a_args, union_args=b_args)
    elif b_origin == typing.Union and a_origin is None:
        return check_union(origin_type=type_a, args=(), union_args=b_args)
    elif a_origin is not None and b_origin is not None:  # TODO: good default case
        return a_origin == b_origin and a_args == b_args
    else:
        raise NotImplementedError(f"Uncatched case: '{type_a}' and '{type_b}'")


class CheckTypeHints(CheckerBaseModel):
    """Represents a client query to check if an array of python types match
    against some other type."""

    types: typing.List[str]
    expected_type: str

    @root_validator()
    def check_valid_types(cls, values):
        memo = from_frame(0)
        expected_type = values["expected_type"]
        failed_indexs: typing.List[typing.Tuple[int, Exception]] = []
        for index, type in enumerate(values["types"]):
            if not check_types(expected_type, type, memo):
                failed_indexs.append(
                    (index, TypeError(f"{type} doesn't match {expected_type}"))
                )
        if len(failed_indexs) > 0:
            raise ValidationError(
                model=CheckTypeHints,
                errors=[
                    ErrorWrapper(exp, loc=("types", str(idx)))
                    for idx, exp in failed_indexs
                ],
            )
        return values
