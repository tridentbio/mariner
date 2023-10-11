"""
Handlers for api/v1/login* endpoints
"""
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm

from mariner.exceptions import UserNotActive, UserNotFound
from mariner.schemas.token import Token
from mariner.users import BasicAuth, authenticate
from mariner.utils.metrics import REQUEST_TIME

router = APIRouter()


@REQUEST_TIME.labels(endpoint="/login/access-token", method="POST").time()
@router.post("/login/access-token", response_model=Token)
def login_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    try:
        token = authenticate(
            basic=BasicAuth.construct(
                username=form_data.username, password=form_data.password
            )
        )
        return token
    except UserNotFound as exc:
        raise HTTPException(
            status_code=400, detail="Incorrect email or password"
        ) from exc
    except UserNotActive as exc:
        raise HTTPException(status_code=400, detail="Inactive user") from exc
