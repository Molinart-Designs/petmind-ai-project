import pytest
from fastapi import HTTPException


def test_ingest_endpoint_returns_201_with_valid_payload(client, sample_ingest_payload):
    response = client.post(
        "/api/v1/ingest",
        json=sample_ingest_payload,
    )

    assert response.status_code == 201

    body = response.json()
    assert body["status"] == "completed"
    assert body["source"] == "demo_batch"
    assert body["documents_received"] == 1
    assert body["documents_processed"] == 1
    assert body["chunks_created"] == 3
    assert body["document_ids"] == ["doc-demo-1"]
    assert "message" in body
    assert "ingested_at" in body


@pytest.mark.no_auth_override
def test_ingest_endpoint_returns_401_when_bearer_token_is_missing(client, sample_ingest_payload):
    response = client.post(
        "/api/v1/ingest",
        json=sample_ingest_payload,
    )

    assert response.status_code == 401
    body = response.json()
    assert "Missing bearer access token" in body["detail"]


@pytest.mark.no_auth_override
def test_ingest_endpoint_returns_401_when_access_token_is_invalid(
    client, sample_ingest_payload, monkeypatch
):
    def _invalid(_token: str):
        raise HTTPException(status_code=401, detail="Invalid access token.")

    monkeypatch.setattr("src.security.auth_jwt.validate_access_token", _invalid)

    response = client.post(
        "/api/v1/ingest",
        json=sample_ingest_payload,
        headers={"Authorization": "Bearer invalid-token"},
    )

    assert response.status_code == 401
    body = response.json()
    assert body["detail"] == "Invalid access token."


@pytest.mark.no_auth_override
def test_ingest_endpoint_returns_403_when_missing_ingest_permission(
    client, sample_ingest_payload, monkeypatch
):
    def _validate(_token: str):
        return {"sub": "auth0|no-ingest", "permissions": ["query:ask"]}

    monkeypatch.setattr("src.security.auth_jwt.validate_access_token", _validate)

    response = client.post(
        "/api/v1/ingest",
        json=sample_ingest_payload,
        headers={"Authorization": "Bearer any-token"},
    )

    assert response.status_code == 403
    body = response.json()
    assert "ingest:write" in body["detail"]


def test_ingest_endpoint_returns_422_for_invalid_payload(client):
    payload = {
        "source": "demo_batch",
        "documents": [],
    }

    response = client.post(
        "/api/v1/ingest",
        json=payload,
    )

    assert response.status_code == 422
