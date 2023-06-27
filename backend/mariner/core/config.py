"""
Configuration for running environment variables.

Shared between API and mariner, on ray and backend instances
Sometimes cause the application to fail when missing an ENV VAR
"""
import secrets
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import (
    AnyHttpUrl,
    BaseSettings,
    EmailStr,
    PostgresDsn,
    root_validator,
    validator,
)


def make_list_from_array_string(v: Union[str, list[str]]):
    """Takes an input that is maybe a string or a list of strings and
    maps it to a list of strings with at least 1 element."""
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",")]
    elif isinstance(v, (list, str)):
        return v
    raise ValueError(v)


class Settings(BaseSettings):
    """Models the environment variables used around the application."""

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
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
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
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
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

    GITHUB_CLIENT_ID: Union[None, str] = None
    GITHUB_CLIENT_SECRET: Union[None, str] = None

    LIGHTNING_LOGS_DIR: str

    @root_validator(allow_reuse=True)
    @classmethod
    def validate_aws_creds(cls, values: Dict[str, Any]) -> Any:
        """
        Validate that AWS credentials are set if AWS_MODE is set to local.
        """
        if values["AWS_MODE"] == "local":
            if not values["AWS_ACCESS_KEY_ID"]:
                raise ValueError("AWS_ACCESS_KEY_ID must be set if AWS_MODE is local")
            if not values["AWS_SECRET_ACCESS_KEY"]:
                raise ValueError(
                    "AWS_SECRET_ACCESS_KEY must be set if AWS_MODE is local"
                )
        elif values["AWS_MODE"] == "sts":
            if bool(values["AWS_ACCESS_KEY_ID"]) or bool(
                values["AWS_SECRET_ACCESS_KEY"]
            ):
                raise ValueError(
                    "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must not be set if AWS_MODE is sts"
                )
        return values

    class Config:
        """Configure the environment variable model to be case sensitive."""

        case_sensitive = True


def get_app_settings() -> Settings:
    """
    Get the application settings.

    Returns:
        Settings: the application settings.
    """
    settings = Settings()
    return settings
