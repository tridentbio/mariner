"""
Handlers for api/v1/oauth* endpoints
"""
import logging
import traceback
from typing import Optional

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
from mariner.exceptions.user_exceptions import UserEmailNotAllowed
from mariner.users import GithubAuth, authenticate

router = APIRouter()
LOG = logging.getLogger(__name__)


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
def receive_github_code(
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    error_uri: Optional[str] = None,
    error_description: Optional[str] = None,
):
    if code and state:
        try:
            token = authenticate(
                github_oauth=GithubAuth.construct(code=code, state=state)
            )
            return RedirectResponse(
                url=f"{settings.WEBAPP_URL}/login?tk={token.access_token}&tk_type={token.token_type}"  # noqa: E501
            )
        except UserEmailNotAllowed:
            return RedirectResponse(
                url=f"{settings.WEBAPP_URL}/login?error=Email not cleared for github oauth"  # noqa: E501
            )
        except UserNotActive:
            return RedirectResponse(
                url=f"{settings.WEBAPP_URL}/login?error=Inactive user"
            )
        except (InvalidGithubCode, InvalidOAuthState):
            return RedirectResponse(
                url=f"{settings.WEBAPP_URL}/login?error=Invalid auth attempt"
            )
        except Exception:
            lines = traceback.format_exc()
            LOG.error("Unexpected auth error: %s", lines)
            return RedirectResponse(
                url=f"{settings.WEBAPP_URL}/login?error=Internal Error"
            )
    elif error or error_description:
        LOG.error("Github auth error: %s: %s", error, error_description)
        return RedirectResponse(url=f"{settings.WEBAPP_URL}/login?error=Internal Error")

    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
