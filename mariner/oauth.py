from sqlalchemy.orm import Session

from mariner.core.config import settings
from mariner.stores.oauth_state_sql import oauth_state_store


def get_github_oauth_url(db: Session):
    state = oauth_state_store.create_state(db, provider="github").state
    redirect_uri = f"{settings.SERVER_HOST}/api/v1/oauth-callback"
    return f"https://github.com/login/oauth/authorize?client_id={settings.GITHUB_CLIENT_ID}&redirect_uri={redirect_uri}&scope=read:user,user:email&state={state}"


provider_url_makers = {"github": get_github_oauth_url}
