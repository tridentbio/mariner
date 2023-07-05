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
from mariner.core.config import get_app_settings
from mariner.exceptions import (
    InvalidGithubCode,
    InvalidOAuthState,
    UserNotActive,
)
from mariner.exceptions.user_exceptions import UserEmailNotAllowed
from mariner.users import GithubAuth, authenticate

router = APIRouter()
LOG = logging.getLogger(__name__)


@router.get("/oauth-providers")
def get_oauth_providers():
    """Endpoint to get list of oauth providers."""
    return oauth.oauth_manager


@router.get("/oauth")
def get_oauth_provider_redirect(provider: str, db: Session = Depends(deps.get_db)):
    """Endpoint to redirect user to provider authentication site.

    Args:
        provider: string of the oauth provider.
        db: database connection.
    """
    if provider not in oauth.oauth_manager:
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No oauth provider named {provider}",
        )
    url = oauth.oauth_manager.get_redirect_uri(provider)
    return RedirectResponse(url=url)


@router.get("/oauth-callback")
def receive_github_code(
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    error_uri: Optional[str] = None,
    error_description: Optional[str] = None,
):
    """Function to handle callback from github with

    Args:
        code: github oauth code
        state: state produced by application previously
        error: error from oauth setup
        error_uri: link to more detailed description of the error.
        error_description: message describing the error

    Raises:
        HTTPException: When the github cod is invalid.
    """
    # TODO, adapt this function to multiple oauth providers. Currently it is only implemented for github.
    if code and state:
        try:
            token = authenticate(
                github_oauth=GithubAuth.construct(code=code, state=state)
            )
            return RedirectResponse(
                url=f"{get_app_settings('webapp').url}/login?tk={token.access_token}&tk_type={token.token_type}"  # noqa: E501
            )
        except UserEmailNotAllowed:
            return RedirectResponse(
                url=f"{get_app_settings('webapp').url}/login?error=Email not cleared for github oauth"  # noqa: E501
            )
        except UserNotActive:
            return RedirectResponse(
                url=f"{get_app_settings('webapp').url}/login?error=Inactive user"
            )
        except (InvalidGithubCode, InvalidOAuthState):
            return RedirectResponse(
                url=f"{get_app_settings('webapp').url}/login?error=Invalid auth attempt"
            )
        except Exception:
            lines = traceback.format_exc()
            LOG.error("Unexpected auth error: %s", lines)
            return RedirectResponse(
                url=f"{get_app_settings('webapp').url}/login?error=Internal Error"
            )
    elif error or error_description:
        LOG.error("Github auth error: %s: %s", error, error_description)
        return RedirectResponse(
            url=f"{get_app_settings('webapp').url}/login?error=Internal Error"
        )

    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
