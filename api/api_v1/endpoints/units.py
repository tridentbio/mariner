from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status

from mariner import units

router = APIRouter()


@router.get("/", response_model=List[units.Unit])
def get_units(q: Optional[str]):
    return units.search_units(q if q is not None else "")


@router.get("/valid", response_model=units.Unit)
def get_is_unit_valid(q: str = Query(...)):
    valid_unit = units.valid_unit(q)
    if not valid_unit:
        raise HTTPException(
            status_code=status.HTTP_204_NO_CONTENT, detail="not a pint unit"
        )
    return valid_unit
