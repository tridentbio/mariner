from functools import lru_cache
from typing import List, Optional

import pint

from app.schemas.api import ApiBaseModel


class Unit(ApiBaseModel):
    name: str
    latex: str


def make_api_unit_from_pint_unit(punit: pint.Unit) -> Unit:
    return Unit(
        name=punit._repr_html_(),
        latex=punit._repr_latex_(),
    )


@lru_cache
def get_units():
    """Gets all units from the registry. There are around 1000 by default"""
    result: List[Unit] = []
    ureg = pint.UnitRegistry()
    for key in ureg:
        try:
            pint_attr = getattr(ureg, key)
            if isinstance(pint_attr, pint.Unit):
                result.append(make_api_unit_from_pint_unit(pint_attr))
        except pint.errors.UndefinedUnitError:
            continue

    return result


def search_units(search: str) -> List[Unit]:
    units = get_units()
    return [unit for unit in units if search.lower() in unit.name.lower()]


def valid_unit(unit: str) -> Optional[Unit]:
    ureg = pint.UnitRegistry()
    try:
        quantity = ureg(unit)
        return make_api_unit_from_pint_unit(quantity.unit)
    except pint.errors.UndefinedUnitError:
        return None
