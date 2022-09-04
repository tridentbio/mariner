from typing import List

from fastapi import APIRouter, HTTPException, Query, status

from app.features.units import units

router = APIRouter()


@router.get("/", response_model=List[units.Unit])
def get_units(q: str = Query(...)):
    return units.search_units(q)


@router.get("/valid", response_model=units.Unit)
def get_is_valid(q: str = Query(...)):
    valid_unit = units.valid_unit(q)
    if not valid_unit:
        raise HTTPException(
            status_code=status.HTTP_204_NO_CONTENT, detail="not a pint unit"
        )
    return valid_unit
