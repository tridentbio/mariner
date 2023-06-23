"""
Handlers for api/v1/units* endpoints
"""
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status

from mariner import units

router = APIRouter()


@router.get("/", response_model=List[units.Unit])
def get_units(q: Optional[str]):
    """Endpoint to get a list of available valid units.

    Args:
        q: query string to filter results
    """
    return units.search_units(q if q is not None else "")


@router.get("/valid", response_model=units.Unit)
def get_is_unit_valid(q: str = Query(...)):
    """Endpoint that checks if a string represents a valid unit.

    Args:
        q: the string to be checked

    Raises:
        HTTPException: When the input string is not a valid unit.
    """
    valid_unit = units.valid_unit(q)
    if not valid_unit:
        raise HTTPException(
            status_code=status.HTTP_204_NO_CONTENT, detail="not a pint unit"
        )
    return valid_unit
