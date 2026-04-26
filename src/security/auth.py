from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from secrets import compare_digest
from typing import Any, Literal

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.config import get_settings
from src.db.models import User
from src.db.session import get_db
from src.security import auth_jwt
from src.security.auth_jwt import PERMISSION_INGEST_WRITE, PERMISSION_QUERY_ASK

api_key_header = APIKeyHeader(
    name=get_settings().api_key_header_name,
    auto_error=False,
)

bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True, slots=True)
class RequestAuthContext:
    """Resolved principal for a protected request."""

    method: Literal["jwt", "api_key"]
    permissions: frozenset[str]
    claims: dict[str, Any] | None
    db_user: User | None


async def require_api_key(
    provided_api_key: str | None = Security(api_key_header),
) -> str:
    """
    Validates the API key sent in the configured header.
    Returns the API key if valid, otherwise raises 401.
    """
    s = get_settings()
    if not provided_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing API key. Expected header: {s.api_key_header_name}",
        )

    if not compare_digest(provided_api_key, s.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    return provided_api_key


def upsert_user_from_claims(session: Session, claims: dict[str, Any]) -> User:
    """
    Creates or updates the local User row keyed by Auth0 subject (`sub`).
    """
    sub = claims.get("sub")
    if not sub or not isinstance(sub, str):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token is missing subject (sub).",
        )

    email = claims.get("email")
    if email is not None and not isinstance(email, str):
        email = str(email)

    full_name = claims.get("name") or claims.get("nickname")
    if full_name is not None and not isinstance(full_name, str):
        full_name = str(full_name)

    now = datetime.now(timezone.utc)
    user = session.execute(select(User).where(User.auth0_sub == sub)).scalar_one_or_none()
    if user is None:
        user = User(
            auth0_sub=sub,
            email=email,
            full_name=full_name,
            last_login_at=now,
        )
        session.add(user)
        session.flush()
        return user

    if email:
        user.email = email
    if full_name:
        user.full_name = full_name
    user.last_login_at = now
    session.flush()
    return user


def get_request_auth_for_query(
    session: Session = Depends(get_db),
    bearer: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    provided_api_key: str | None = Security(api_key_header),
) -> RequestAuthContext:
    s = get_settings()
    token = bearer.credentials if bearer and bearer.scheme.lower() == "bearer" else None

    if token:
        claims = auth_jwt.validate_access_token(token)
        permissions = frozenset(auth_jwt.parse_permissions(claims))
        if PERMISSION_QUERY_ASK not in permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permission: {PERMISSION_QUERY_ASK}",
            )
        user = upsert_user_from_claims(session, claims)
        return RequestAuthContext(
            method="jwt",
            permissions=permissions,
            claims=claims,
            db_user=user,
        )

    if s.legacy_api_key_fallback_enabled:
        if not provided_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing credentials: Bearer token or legacy API key.",
            )
        if not compare_digest(provided_api_key, s.api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key.",
            )
        return RequestAuthContext(
            method="api_key",
            permissions=frozenset({PERMISSION_QUERY_ASK, PERMISSION_INGEST_WRITE}),
            claims=None,
            db_user=None,
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing bearer access token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_request_auth_for_ingest(
    session: Session = Depends(get_db),
    bearer: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    provided_api_key: str | None = Security(api_key_header),
) -> RequestAuthContext:
    s = get_settings()
    token = bearer.credentials if bearer and bearer.scheme.lower() == "bearer" else None

    if token:
        claims = auth_jwt.validate_access_token(token)
        permissions = frozenset(auth_jwt.parse_permissions(claims))
        if PERMISSION_INGEST_WRITE not in permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permission: {PERMISSION_INGEST_WRITE}",
            )
        user = upsert_user_from_claims(session, claims)
        return RequestAuthContext(
            method="jwt",
            permissions=permissions,
            claims=claims,
            db_user=user,
        )

    if s.legacy_api_key_fallback_enabled:
        if not provided_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing credentials: Bearer token or legacy API key.",
            )
        if not compare_digest(provided_api_key, s.api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key.",
            )
        return RequestAuthContext(
            method="api_key",
            permissions=frozenset({PERMISSION_QUERY_ASK, PERMISSION_INGEST_WRITE}),
            claims=None,
            db_user=None,
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing bearer access token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
