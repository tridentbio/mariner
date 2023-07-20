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

<<<<<<< HEAD
    API_V1_STR: str = "/api/v1"
    AUTHENTICATION_SECRET_KEY: str = secrets.token_urlsafe(32)
    DEPLOYMENT_URL_SIGNATURE_SECRET_KEY: str = secrets.token_urlsafe(32)

    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8

    DEPLOYMENT_IDLE_TIME: int = 60 * 10  # 10 minutes

    SERVER_NAME: str
    SERVER_HOST: AnyHttpUrl
    WEBAPP_URL: str
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []
    ALLOWED_GITHUB_AUTH_EMAILS: List[EmailStr] = []

    @validator("BACKEND_CORS_ORIGINS", pre=True, allow_reuse=True)
    def assemble_cors_origins(
        cls, v: Union[str, List[str]]
    ) -> Union[List[str], str]:
        """Parses the string of allowed cors origins into a list of strings.

        Returns:
            list of strings from the array string.
        """
        return make_list_from_array_string(v)

    @validator("ALLOWED_GITHUB_AUTH_EMAILS", pre=True, allow_reuse=True)
    def assemble_allowed_github_auth_emails(
        cls, v: Union[str, List[str]]
    ) -> Union[List[str], str]:
        """Parses the string of allowed cors origins into a list of strings.

        Returns:
            list of strings from the array string.
        """
        return make_list_from_array_string(v)

    PROJECT_NAME: str
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    SQLALCHEMY_DATABASE_URI: Optional[PostgresDsn] = None

    @validator("SQLALCHEMY_DATABASE_URI", pre=True, allow_reuse=True)
    def assemble_db_connection(
        cls, v: Optional[str], values: Dict[str, Any]
    ) -> Any:
        """Assembles the SQLALCHEMY_DATABASE_URI directly from var with same name in .env
        or from .env POSTGRES_* variables.

        Args:
            v: SQLALCHEMY_DATABASE_URI if is present on the env file.
            values: dictionary of unvalidated object.

        Returns:
            A PostgresDsn object with connection values.
        """
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql",
            user=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_SERVER") or "localhost",
            path=f"/{values.get('POSTGRES_DB') or ''}",
        )

    EMAILS_ENABLED: bool = False
    EMAIL_TEST_USER: EmailStr = "test@example.com"  # type: ignore

    AWS_MODE: Literal["sts", "local"] = "local"
    AWS_ACCESS_KEY_ID: Union[None, str] = None
    AWS_SECRET_ACCESS_KEY: Union[None, str] = None
    AWS_REGION: str = "us-east-1"
    AWS_DATASETS: str = "dev-matiner-datasets"
    AWS_MODELS: str = "dev-matiner-datasets"

    RAY_ADDRESS: str = "ray://ray-head:10001"
    APPLICATION_SECRET: str = "replaceme"
    APPLICATION_CHUNK_SIZE: int = 1024

=======
>>>>>>> origin/develop
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
    scope: str
    allowed_emails: Union[None, list[str]] = None
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

        # populate missing_envs with missing required environment variables
        # from oauth providers
        missing_envs = []
        required_fields = [
            "CLIENT_ID",
            "CLIENT_SECRET",
            "NAME",
            "AUTHORIZATION_URL",
            "SCOPE",
        ]
        for provider in providers:
            for required_field in required_fields:
                if auth_env.get(f"{provider}_{required_field}") is None:
                    missing_envs.append(f"OAUTH_{provider}_{required_field}")
        # Raise error if missing_envs is not empty
        if missing_envs:
            raise ValueError(
                f"Missing required environment variables for OAuth providers: {missing_envs}"
            )

        # Create auth settings for each provider
        self.auth = AuthSettingsDict(
            __root__={
                provider.lower(): AuthSettings(
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
def get_app_settings(
    name: Union[str, None] = None, use_cache: bool = False
) -> SettingsV2:
    ...


@overload
def get_app_settings(name: Literal["server"], use_cache=False) -> ServerSettings:
    ...


@overload
def get_app_settings(name: Literal["webapp"], use_cache=False) -> WebappSettings:
    ...


@overload
def get_app_settings(name: Literal["auth"], use_cache=False) -> AuthSettingsDict:
    ...


@overload
def get_app_settings(name: Literal["secrets"], use_cache=False) -> SecretEnv:
    ...


@overload
def get_app_settings(name: Literal["services"], use_cache=False) -> ServicesEnv:
    ...


@overload
def get_app_settings(name: Literal["package"], use_cache=False) -> Package:
    ...


@overload
def get_app_settings(name: Literal["test"], use_cache=False) -> QA_Test_Settings:
    ...


@overload
def get_app_settings(name: Literal["tenant"], use_cache=False) -> TenantSettings:
    ...


def get_app_settings(name: Union[str, None] = None, use_cache=True):
    """
    Get the application settings.

    Returns:
        Settings: the application settings.
    """
    if use_cache:
        settings = _load_settings()
    else:
        settings = _load_settings.__wrapped__()
    if name is None:
        return settings
    else:
        return getattr(settings, name)
