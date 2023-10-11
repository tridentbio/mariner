"""
Handlers for api/v1/oauth* endpoints
"""
import logging
import traceback
from typing import List

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import RedirectResponse

from mariner import oauth
from mariner.core.config import get_app_settings
from mariner.exceptions import UserNotActive
from mariner.exceptions.user_exceptions import UserEmailNotAllowed
from mariner.users import authenticate
from mariner.utils.metrics import REQUEST_TIME

router = APIRouter()
LOG = logging.getLogger(__name__)


@REQUEST_TIME.labels(endpoint="/oauth-providers", method="GET").time()
@router.get("/oauth-providers", response_model=List[oauth.Provider])
def get_oauth_providers():
    """Endpoint to get list of oauth providers."""
    return oauth.oauth_manager.get_providers()


@REQUEST_TIME.labels(endpoint="/oauth", method="GET").time()
@router.get("/oauth")
def get_oauth_provider_redirect(provider: str):
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


@REQUEST_TIME.labels(endpoint="/oauth-callback", method="GET").time()
@router.get("/oauth-callback")
def get_oauth_callback(request: Request):
    """Function to handle callback from oauth provider.

    The function will authenticate the user and redirect to the webapp with
    the token as a query parameter. Handles the request as an error if there
    is some query parameter that starts with "error".

    Raises:
        HTTPException: When the provider code is invalid.
    """
    if not request.query_params:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No query parameters",
        )
    state = request.query_params.get("state")
    errors = {
        query_key: query_value
        for query_key, query_value in request.query_params.items()
        if query_key.startswith("error")
    }
    rest = {
        query_key: query_value
        for query_key, query_value in request.query_params.items()
        if query_key != "state"
    }

    if state and not errors:
        try:
            token = authenticate(
                provider_oauth={"state": state, **rest},
            )
            return RedirectResponse(
                url=f"{get_app_settings('webapp').url}/login?tk={token.access_token}&tk_type={token.token_type}"  # noqa: E501
            )
        except UserEmailNotAllowed:
            return RedirectResponse(
                url=f"{get_app_settings('webapp').url}/login?error=Email not cleared for provider oauth"  # noqa: E501
            )
        except UserNotActive:
            return RedirectResponse(
                url=f"{get_app_settings('webapp').url}/login?error=Inactive user"
            )
        except Exception:  # pylint: disable=W0718
            lines = traceback.format_exc()
            LOG.error("Unexpected auth error: %s", lines)
            return RedirectResponse(
                url=f"{get_app_settings('webapp').url}/login?error=Internal Error"
            )
    elif errors:
        LOG.error("OAuth provider error: %r", errors)
        return RedirectResponse(
            url=f"{get_app_settings('webapp').url}/login?error=Internal Error"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid auth attempt",
        )
