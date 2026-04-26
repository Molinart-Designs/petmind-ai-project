"""
Auth0 access token validation (RS256 + JWKS) and RBAC permission parsing.

Validates issuer, audience, signature, and expiration via PyJWT.
Permissions are read from the standard Auth0 RBAC claim: `permissions` (array of strings).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import jwt
from fastapi import HTTPException, status
from jwt import PyJWKClient

from src.core.config import get_settings

PERMISSION_QUERY_ASK = "query:ask"
PERMISSION_INGEST_WRITE = "ingest:write"


def _normalized_domain(domain: str) -> str:
    d = domain.strip()
    if d.startswith("https://"):
        d = d[8:]
    if d.startswith("http://"):
        d = d[7:]
    return d.rstrip("/")


def _jwks_uri() -> str:
    s = get_settings()
    host = _normalized_domain(s.auth0_domain)
    if not host:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Auth0 is not configured: AUTH0_DOMAIN is missing or empty.",
        )
    return f"https://{host}/.well-known/jwks.json"


@lru_cache(maxsize=1)
def _jwks_client() -> PyJWKClient:
    return PyJWKClient(_jwks_uri())


def clear_jwks_cache() -> None:
    """Test hook: reset cached JWKS client (e.g. after settings change)."""
    clear = getattr(_jwks_client, "cache_clear", None)
    if callable(clear):
        clear()


def parse_permissions(claims: dict[str, Any]) -> list[str]:
    raw = claims.get("permissions")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [p for p in raw.split() if p]
    if isinstance(raw, (list, tuple)):
        return [str(p) for p in raw if isinstance(p, str)]
    return []


def validate_access_token(token: str) -> dict[str, Any]:
    """
    Decode and validate an Auth0-issued JWT access token.

    Checks signature (JWKS), algorithms, audience, issuer, and expiration.
    """
    if not token or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing access token.",
        )

    s = get_settings()
    issuer = (s.auth0_issuer or "").strip()
    audience = (s.auth0_audience or "").strip()
    if not issuer or not audience:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Auth0 JWT validation is not configured (AUTH0_ISSUER / AUTH0_AUDIENCE).",
        )

    algorithms = s.auth0_algorithm_list
    try:
        signing_key = _jwks_client().get_signing_key_from_jwt(token)
        payload: dict[str, Any] = jwt.decode(
            token,
            signing_key.key,
            algorithms=algorithms,
            audience=audience,
            issuer=issuer,
            options={
                "require": ["exp", "sub"],
                "verify_aud": True,
                "verify_iss": True,
            },
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token has expired.",
        ) from exc
    except jwt.InvalidAudienceError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token audience.",
        ) from exc
    except jwt.InvalidIssuerError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token issuer.",
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token.",
        ) from exc

    return payload
