"""
Units service.

Uses pint to get and validate units and quantities
"""
import re
from functools import lru_cache
from typing import Dict, List, Literal, NewType, Optional, Tuple, Union

import pint

from mariner.schemas.api import ApiBaseModel


class Unit(ApiBaseModel):
    """Models a unit that can be displayed on markup or text."""

    name: str
    latex: str


def make_api_unit_from_pint_unit(punit: Union[pint.Quantity, pint.Unit]) -> Unit:
    """Maps pint unit to the mariner's unit object.

    Args:
        punit: the unit from pint.

    Returns:
        Unit
    """
    return Unit(
        name=punit._repr_html_(),
        latex=punit._repr_latex_(),
    )


@lru_cache
def get_units():
    """Gets all units from the registry. There are around 1000 by default"""
    result: List[Unit] = []
    _used = {}
    ureg = pint.UnitRegistry()
    for key in ureg:
        try:
            pint_attr = getattr(ureg, key)
            if (
                isinstance(pint_attr, pint.Unit)
                and pint_attr._repr_html_() not in _used
            ):
                result.append(make_api_unit_from_pint_unit(pint_attr))
                _used[pint_attr._repr_html_()] = True
        except pint.errors.UndefinedUnitError:
            continue

    return result


def search_units(search: str = "") -> List[Unit]:
    """Get a filtered list of accepted units.

    Returns the list of units accepted by this filtering those that
    contains search in it's name attribute.

    Args:
        search: String to search the units

    Returns:
        A list of units that match the search
    """
    units = get_units()
    return [unit for unit in units if search.lower() in unit.name.lower()]


CheckedLogarithmicUnit = NewType(
    "base, inner_unit, rest_before, rest_after",
    Tuple[Optional[str], Optional[str], Optional[str], Optional[str]],
)


def check_logarithmic_unit(unit: str) -> CheckedLogarithmicUnit:
    """Checks if the unit is a logarithmic unit

    Returns None if it's not a logarithmic unit, otherwise
    Returns the base, the inner unit and the rest of the unit

    Args:
        unit: String name of the unit to be validated

    Returns:
        base, inner_unit, rest_bef_str, rest_aft_str

    Raises:
        ValueError: When the logarithmic is invalid
    """
    search = re.search(
        r"([^\(]*)?(log|ln)(\d*)\(([^\)\(]+|(?:.+\([^\)]+\))+[^\)]*)\)(.*)",
        unit.strip(),
    )
    if search:
        rest_bef_str, log_name, base, inner_str, rest_aft_str = search.groups()
        if log_name == "ln":
            if base:
                raise ValueError(f"ln does not accept a base")
            return "e", inner_str, rest_bef_str, rest_aft_str
        else:
            return base if base else "10", inner_str, rest_bef_str, rest_aft_str
    else:
        return None, None, None, None


def parse_rest_unit(
    rest_unit: Optional[str], before: bool
) -> Tuple[Optional[str], Optional[str]]:
    """Parses the rest of the unit after the logarithmic unit

    Expected format: <operation?><unit>
    operation: *, /, **, '' (default: *)

    Raises:
        ValueError: When the operation is invalid

    Returns:
        operation, unit
    """
    regex = r"([\w\(\)]+)(\W?)" if before else r"(\W?)(.+)"

    search = re.search(regex, rest_unit.strip() or "")
    if search:
        operation, operator = search.groups()[::-1] if before else search.groups()
        if operation not in ["*", "/", "**", ""]:
            raise ValueError(f"Invalid operation: {operation}")

        return operation or "*", operator
    else:
        return None, None


def generate_serialized_unit(unit: str) -> Dict[Literal["name", "latex"], str]:
    """Generates a serialized unit from a string or raises an error if it's invalid

    First, it checks if the unit is a logarithmic unit, if it is, it parses it recursively

    Args:
        unit: String name of the unit to be validated

    Returns:
        Dict with the name and latex of the unit
        name: String name of the unit
        latex: String latex of the unit

    Raises:
        ValueError: When the unit is invalid
    """

    logarithmic_base, inner_unit_str, *rest = check_logarithmic_unit(unit)

    if logarithmic_base:
        inner_unit = generate_serialized_unit(inner_unit_str)
        if not inner_unit:
            raise ValueError(f"Invalid unit inside a logarithmic: {inner_unit}")

        log, log_ = (
            (f"log{logarithmic_base}", f"log_{logarithmic_base} ")
            if logarithmic_base != "e"
            else ("ln", "ln ")
        )
        name = f'{log}({inner_unit["name"]})'
        latex = f'{log_}({inner_unit["latex"]})'

        for i, rest_str in enumerate(rest):
            operation, operator = parse_rest_unit(rest_str, i == 0)
            if operator:
                operator_unit = generate_serialized_unit(operator)
                if i == 0:  # before
                    name = f'{operator_unit["name"]} {operation} {name}'
                    latex = f'{operator_unit["latex"]} {operation} {latex}'
                else:  # after
                    name = f'{name} {operation} {operator_unit["name"]}'
                    latex = f'{latex} {operation} {operator_unit["latex"]}'
        return {
            "name": name,
            "latex": latex,
        }

    try:
        ureg = pint.UnitRegistry()
        quantity = ureg(unit)
        return {
            "name": quantity._repr_html_(),
            "latex": quantity._repr_latex_(),
        }
    except:
        ValueError(f"Invalid unit: {unit}")


def valid_unit(unit: str) -> Optional[Unit]:
    """Checks if the unit is a valid name for a unit.

    Uses the pint library under the hood to normalize different names for the same unit.

    Args:
        unit: String name of the unit to be validated.

    Returns:
        Unit if unit is valid, None otherwise.
    """
    try:
        serialized_unity = generate_serialized_unit(unit)
        return Unit(**serialized_unity)
    except (ValueError, TypeError):
        return None
