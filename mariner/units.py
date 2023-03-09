"""
Units service.

Uses pint to get and validate units and quantities
"""
from functools import lru_cache
from typing import List, Optional, Union

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


def valid_unit(unit: str) -> Optional[Unit]:
    """Checks if the unit is a valid name for a unit.

    Uses the pint library under the hood to normalize different names for the same unit.

    Args:
        unit: String name of the unit to be validated.

    Returns:
        Unit if unit is valid, None otherwise.
    """
    ureg = pint.UnitRegistry()
    try:
        quantity = ureg(unit)
        return make_api_unit_from_pint_unit(quantity)
    except pint.errors.UndefinedUnitError:
        return None
