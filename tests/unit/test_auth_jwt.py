"""Unit tests for Auth0 JWT validation helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from src.core.config import get_settings
from src.security.auth_jwt import (
    PERMISSION_INGEST_WRITE,
    PERMISSION_QUERY_ASK,
    clear_jwks_cache,
    parse_permissions,
    validate_access_token,
)


@pytest.fixture
def auth0_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH0_DOMAIN", "tenant.example.com")
    monkeypatch.setenv("AUTH0_AUDIENCE", "https://petmind.api/")
    monkeypatch.setenv("AUTH0_ISSUER", "https://tenant.example.com/")
    monkeypatch.setenv("AUTH0_ALGORITHMS", "RS256")
    get_settings.cache_clear()
    clear_jwks_cache()
    yield
    get_settings.cache_clear()
    clear_jwks_cache()


def _rsa_token(
    *,
    private_key: object,
    claims: dict,
) -> str:
    return jwt.encode(
        claims,
        private_key,
        algorithm="RS256",
        headers={"kid": "unit-test-kid"},
    )


@pytest.fixture
def rsa_pair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    return private_key, public_pem


def test_parse_permissions_from_list():
    claims = {"permissions": ["query:ask", "ingest:write"]}
    perms = parse_permissions(claims)
    assert perms == ["query:ask", "ingest:write"]


def test_parse_permissions_from_string():
    claims = {"permissions": "query:ask ingest:write"}
    perms = parse_permissions(claims)
    assert perms == ["query:ask", "ingest:write"]


def test_parse_permissions_missing_returns_empty():
    assert parse_permissions({}) == []


def test_parse_permissions_ignores_non_strings_in_list():
    claims = {"permissions": ["query:ask", 123, None]}
    assert parse_permissions(claims) == ["query:ask"]


def test_validate_access_token_success(monkeypatch: pytest.MonkeyPatch, auth0_env, rsa_pair):
    private_key, public_pem = rsa_pair
    now = datetime.now(timezone.utc)
    claims = {
        "sub": "auth0|unit-user",
        "iss": "https://tenant.example.com/",
        "aud": "https://petmind.api/",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=10)).timestamp()),
        "permissions": [PERMISSION_QUERY_ASK],
    }
    token = _rsa_token(private_key=private_key, claims=claims)

    mock_jwks = MagicMock()
    mock_jwks.get_signing_key_from_jwt.return_value = MagicMock(key=public_pem)
    monkeypatch.setattr("src.security.auth_jwt._jwks_client", lambda: mock_jwks)

    payload = validate_access_token(token)
    assert payload["sub"] == "auth0|unit-user"
    assert PERMISSION_QUERY_ASK in parse_permissions(payload)


def test_validate_access_token_expired(monkeypatch: pytest.MonkeyPatch, auth0_env, rsa_pair):
    private_key, public_pem = rsa_pair
    now = datetime.now(timezone.utc)
    claims = {
        "sub": "auth0|unit-user",
        "iss": "https://tenant.example.com/",
        "aud": "https://petmind.api/",
        "iat": int((now - timedelta(hours=2)).timestamp()),
        "exp": int((now - timedelta(hours=1)).timestamp()),
    }
    token = _rsa_token(private_key=private_key, claims=claims)

    mock_jwks = MagicMock()
    mock_jwks.get_signing_key_from_jwt.return_value = MagicMock(key=public_pem)
    monkeypatch.setattr("src.security.auth_jwt._jwks_client", lambda: mock_jwks)

    with pytest.raises(HTTPException) as exc_info:
        validate_access_token(token)
    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail.lower()


def test_validate_access_token_wrong_audience(monkeypatch: pytest.MonkeyPatch, auth0_env, rsa_pair):
    private_key, public_pem = rsa_pair
    now = datetime.now(timezone.utc)
    claims = {
        "sub": "auth0|unit-user",
        "iss": "https://tenant.example.com/",
        "aud": "https://other.api/",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=10)).timestamp()),
    }
    token = _rsa_token(private_key=private_key, claims=claims)

    mock_jwks = MagicMock()
    mock_jwks.get_signing_key_from_jwt.return_value = MagicMock(key=public_pem)
    monkeypatch.setattr("src.security.auth_jwt._jwks_client", lambda: mock_jwks)

    with pytest.raises(HTTPException) as exc_info:
        validate_access_token(token)
    assert exc_info.value.status_code == 401
    assert "audience" in exc_info.value.detail.lower()


def test_validate_access_token_wrong_issuer(monkeypatch: pytest.MonkeyPatch, auth0_env, rsa_pair):
    private_key, public_pem = rsa_pair
    now = datetime.now(timezone.utc)
    claims = {
        "sub": "auth0|unit-user",
        "iss": "https://evil.example.com/",
        "aud": "https://petmind.api/",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=10)).timestamp()),
    }
    token = _rsa_token(private_key=private_key, claims=claims)

    mock_jwks = MagicMock()
    mock_jwks.get_signing_key_from_jwt.return_value = MagicMock(key=public_pem)
    monkeypatch.setattr("src.security.auth_jwt._jwks_client", lambda: mock_jwks)

    with pytest.raises(HTTPException) as exc_info:
        validate_access_token(token)
    assert exc_info.value.status_code == 401
    assert "issuer" in exc_info.value.detail.lower()


def test_validate_access_token_blank_raises_401(auth0_env):
    with pytest.raises(HTTPException) as exc_info:
        validate_access_token("   ")
    assert exc_info.value.status_code == 401


def test_validate_access_token_missing_server_config_raises_500(monkeypatch: pytest.MonkeyPatch, auth0_env):
    monkeypatch.setenv("AUTH0_AUDIENCE", "")
    get_settings.cache_clear()

    with pytest.raises(HTTPException) as exc_info:
        validate_access_token("not-used")
    assert exc_info.value.status_code == 500

    monkeypatch.setenv("AUTH0_AUDIENCE", "https://petmind.api/")
    get_settings.cache_clear()


def test_permission_constants():
    assert PERMISSION_QUERY_ASK == "query:ask"
    assert PERMISSION_INGEST_WRITE == "ingest:write"
