"""
Configuration for running environment variables.

Shared between API and mariner, on ray and backend instances
Sometimes cause the application to fail when missing an ENV VAR
"""
import secrets
from typing import Any, Dict, List, Optional, Union

from pydantic import AnyHttpUrl, BaseSettings, EmailStr, PostgresDsn, validator


def make_list_from_array_string(v: Union[str, list[str]]):
    """Takes an input that is maybe a string or a list of strings and
    maps it to a list of strings with at least 1 element."""
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",")]
    elif isinstance(v, (list, str)):
        return v
    raise ValueError(v)


# Make settings be the aggregation of several settings
# Possibly using multi-inheritance
class Settings(BaseSettings):
    """Models the environment variables used around the application."""

    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8

    SERVER_NAME: str
    SERVER_HOST: AnyHttpUrl
    WEBAPP_URL: str
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []
    ALLOWED_GITHUB_AUTH_EMAILS: List[EmailStr] = []

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        """Parses the string of allowed cors origins into a list of strings.

        Returns:
            list of strings from the array string.
        """
        return make_list_from_array_string(v)

    @validator("ALLOWED_GITHUB_AUTH_EMAILS", pre=True)
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

    @validator("SQLALCHEMY_DATABASE_URI", pre=True)
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

    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "us-east-1"
    AWS_DATASETS: str = "dev-matiner-datasets"
    AWS_MODELS: str = "dev-matiner-datasets"

    RAY_ADDRESS: str = "ray://ray-head:10001"
    APPLICATION_SECRET: str
    APPLICATION_CHUNK_SIZE: int = 1024

    GITHUB_CLIENT_ID: str
    GITHUB_CLIENT_SECRET: str

    LIGHTNING_LOGS_DIR: str

    class Config:
        """Configure the environment variable model to be case sensitive."""

        case_sensitive = True


settings = Settings()