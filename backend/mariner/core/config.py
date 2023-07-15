"""            
Configuration for running environment variables.


Shared between API and mariner, on ray and backend instances
Sometimes cause the application to fail when missing an ENV VAR
"""

import functools
import os
from typing import Any, Dict, Literal, Optional, Union, overload

import toml
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    BaseSettings,
    PostgresDsn,
    root_validator,
)


class Settings(BaseSettings):
    """Models the environment variables used around the application."""

    GITHUB_CLIENT_ID: Union[None, str] = None
    GITHUB_CLIENT_SECRET: Union[None, str] = None


class ServerSettings(BaseModel):
    """
    Configures server parameters.
    """

    host: AnyHttpUrl
    cors: list[AnyHttpUrl] = []
    access_token_expire_minutes: int = 60 * 24 * 8
    deployment_idle_time: int = 60 * 10  # 10 minutes
    application_chunk_size: int = 1024


class WebappSettings(BaseModel):
    """
    Configures webapp parameters.
    """

    url: str


class AuthSettings(BaseModel):
    """
    Configures authentication parameters.

    If allowed_emails is present, it is used to restrict access to the application.

    Attributes:
        client_id: The client id of the OAuth application.
        client_secret: The client secret of the OAuth application.
        authorization_url: The url to redirect the user to for authentication.
        allowed_emails: A list of emails that are allowed to authenticate.
        scope: The scope of the OAuth application.
        logo_url: The url to get the provider's logo.
    """

    client_id: str
    client_secret: str
    name: str
    authorization_url: str
    allowed_emails: Union[None, list[str]] = None
    scope: Union[str, None] = None
    logo_url: Union[str, None] = None
    jwks_url: Union[str, None] = None
    token_url: Union[str, None] = None


class AuthSettingsDict(BaseModel):
    """
    Holds the authentication settings as a dictionary.
    """

    __root__: Dict[str, AuthSettings]


class TenantSettings(BaseModel):
    """
    Configures tenant parameters.
    """

    name: str


class SecretEnv(BaseSettings):
    """
    Configures secret parameters.

    These are loaded from the environment.
    """

    authentication_secret_key: str
    deployment_url_signature_secret_key: str
    application_secret: str
    aws_mode: Literal["sts", "local"] = "local"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    aws_datasets: str = "dev-matiner-datasets"
    aws_models: str = "dev-matiner-datasets"

    @root_validator(allow_reuse=True)
    @classmethod
    def validate_aws_creds(cls, values: Dict[str, Any]) -> Any:
        """
        Validate that AWS credentials are set if aws_mode is set to local.
        """
        if values.get("aws_mode") == "local":
            if not values.get("aws_access_key_id"):
                raise ValueError("aws_access_key_id must be set if aws_mode is local")
            if not values.get("aws_secret_access_key"):
                raise ValueError(
                    "aws_secret_access_key must be set if aws_mode is local"
                )
        elif values.get("aws_mode") == "sts":
            if values.get("aws_access_key_id", False) or values.get(
                "aws_secret_access_key", False
            ):
                raise ValueError(
                    "aws_access_key_id and aws_secret_access_key must not be set if aws_mode is sts"
                )
        return values


class ServicesEnv(BaseSettings):
    """
    Configures service parameters.

    These are loaded from the environment.
    """

    postgres_uri: PostgresDsn
    ray_address: str
    mlflow_tracking_uri: str


class Package(BaseModel):
    """
    Configures package parameters.

    These are loaded from pyproject.toml.
    """

    name: str
    version: str
    description: str
    authors: list[str]


class QA_Test_Settings(BaseModel):  # pylint: disable=C0103
    """
    Configures the QA test parameters.
    """

    email_test_user: str = "test@domain.com"


class SettingsV2:
    """
    Loads settings from conf.toml, pyproject.toml and environment variables.
    """

    server: ServerSettings
    webapp: WebappSettings
    auth: AuthSettingsDict
    tenant: TenantSettings
    secrets: SecretEnv
    services: ServicesEnv
    package: Package
    test: QA_Test_Settings

    def __init__(self, pyproject_path="pyproject.toml"):
        # load attributes from env
        self.secrets = SecretEnv()  # type: ignore
        self.services = ServicesEnv()  # type: ignore

        # load other attributes from conf.toml
        self.server = ServerSettings(
            host=AnyHttpUrl(
                os.getenv("SERVER_HOST", "http://localhost:8000"), scheme="https"
            ),
            cors=[
                AnyHttpUrl(url, scheme="https")
                for url in os.getenv("SERVER_CORS", "").split(",")
            ],
        )
        self.webapp = WebappSettings(
            url=os.getenv("WEBAPP_URL", "http://localhost:3000")
        )
        self.test = QA_Test_Settings()

        # Get environment variables starting with AUTH
        auth_env = {k[6:]: v for k, v in os.environ.items() if k.startswith("OAUTH")}

        # Get array of providers
        providers = list(set([k.split("_")[0] for k in auth_env.keys()]))

        def split_if_string(string: Union[None, str]):
            if isinstance(string, str):
                return string.split(",")
            return string

        # Create auth settings for each provider
        self.auth = AuthSettingsDict(
            __root__={
                provider: AuthSettings(
                    client_id=auth_env[f"{provider}_CLIENT_ID"],
                    client_secret=auth_env[f"{provider}_CLIENT_SECRET"],
                    name=auth_env[f"{provider}_NAME"],
                    authorization_url=auth_env[f"{provider}_AUTHORIZATION_URL"],
                    allowed_emails=split_if_string(
                        auth_env.get(f"{provider}_ALLOWED_EMAILS", None)
                    ),
                    scope=auth_env.get(f"{provider}_SCOPE", None),
                    logo_url=auth_env.get(f"{provider}_LOGO_URL", None),
                    jwks_url=auth_env.get(f"{provider}_JWKS_URL", None),
                    token_url=auth_env.get(f"{provider}_TOKEN_URL", None),
                )
                for provider in providers
            }
        )

        package_toml = toml.load(pyproject_path)
        self.package = Package.parse_obj(package_toml["tool"]["poetry"])


@functools.cache
def _load_settings() -> SettingsV2:
    return SettingsV2()


@overload
def get_app_settings() -> SettingsV2:
    ...


@overload
def get_app_settings(name: Literal["server"]) -> ServerSettings:
    ...


@overload
def get_app_settings(name: Literal["webapp"]) -> WebappSettings:
    ...


@overload
def get_app_settings(name: Literal["auth"]) -> AuthSettingsDict:
    ...


@overload
def get_app_settings(name: Literal["secrets"]) -> SecretEnv:
    ...


@overload
def get_app_settings(name: Literal["services"]) -> ServicesEnv:
    ...


@overload
def get_app_settings(name: Literal["package"]) -> Package:
    ...


@overload
def get_app_settings(name: Literal["test"]) -> QA_Test_Settings:
    ...


@overload
def get_app_settings(name: Literal["tenant"]) -> TenantSettings:
    ...


def get_app_settings(name: Union[str, None] = None):
    """
    Get the application settings.

    Returns:
        Settings: the application settings.
    """
    settings = _load_settings()
    if name is None:
        return settings
    else:
        return getattr(settings, name)
