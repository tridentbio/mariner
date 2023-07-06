"""            
Configuration for running environment variables.


Shared between API and mariner, on ray and backend instances
Sometimes cause the application to fail when missing an ENV VAR
"""

import functools
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

    name: str
    host: AnyHttpUrl
    domain: str
    cors: list[AnyHttpUrl]
    access_token_expire_minutes: int = 60 * 24 * 8
    deployment_idle_time: int = 60 * 10  # 10 minutes
    api_v1_str: str = "/api/v1"
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
    authorization_url: str
    allowed_emails: Union[None, list[str]] = None
    scope: Union[str, None] = None
    logo_url: Union[str, None] = None


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
        print(values)
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

    def __init__(self, pyproject_path="pyproject.toml", configuration_path="conf.toml"):
        # load attributes from env
        self.secrets = SecretEnv()  # type: ignore
        self.services = ServicesEnv()  # type: ignore

        # load other attributes from conf.toml
        configuration = toml.load(configuration_path)
        self.server = ServerSettings.parse_obj(configuration["server"])
        self.webapp = WebappSettings.parse_obj(configuration["webapp"])
        self.auth = AuthSettingsDict.parse_obj(configuration["auth"])
        self.tenant = TenantSettings.parse_obj(configuration["tenant"])
        self.test = QA_Test_Settings.parse_obj(configuration["test"])

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
