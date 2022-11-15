from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from api import deps
from mariner import oauth
from mariner.core.config import settings
from mariner.exceptions import (
    InvalidGithubCode,
    InvalidOAuthState,
    UserNotActive,
)
from mariner.users import GithubAuth, authenticate

router = APIRouter()


@router.get("/oauth")
def get_oauth_provider_redirect(provider: str, db: Session = Depends(deps.get_db)):
    if provider not in oauth.provider_url_makers:
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No oauth provider named {provider}",
        )
    make_oauth_url = oauth.provider_url_makers[provider]
    return RedirectResponse(url=make_oauth_url(db))


@router.get("/oauth-callback")
def receive_github_code(code: str, state: str):
    try:
        token = authenticate(github_oauth=GithubAuth.construct(code=code, state=state))
        return RedirectResponse(
            url=f"{settings.WEBAPP_URL}/login?tk={token.access_token}&tk_type={token.token_type}"  # noqa E501
        )
    except InvalidGithubCode:
        return HTTPException(status_code=400, detail="Invalid github code")
    except InvalidOAuthState:
        return HTTPException(status_code=400, detail="Invalid state")
    except UserNotActive:
        return HTTPException(status_code=400, detail="Inactive user")
