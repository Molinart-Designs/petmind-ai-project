from secrets import compare_digest

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.core.config import settings

api_key_header = APIKeyHeader(
    name=settings.api_key_header_name,
    auto_error=False,
)


async def require_api_key(
    provided_api_key: str | None = Security(api_key_header),
) -> str:
    """
    Validates the API key sent in the configured header.
    Returns the API key if valid, otherwise raises 401.
    """
    if not provided_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing API key. Expected header: {settings.api_key_header_name}",
        )

    if not compare_digest(provided_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    return provided_api_key