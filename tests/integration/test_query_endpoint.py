import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.orchestrator import get_orchestrator
from src.security.auth import get_request_auth_for_query

from tests import conftest as conftest_mod
from tests.conftest import FakeOrchestrator


class _LeakyReviewDraftOrchestrator(FakeOrchestrator):
    """Ensures the HTTP layer never forwards internal review text to clients."""

    async def answer(self, payload):  # noqa: ANN001
        base = await super().answer(payload)
        return {**base, "review_draft": "must not appear in HTTP JSON"}


def test_query_endpoint_returns_200_with_valid_payload(client, sample_query_payload):
    response = client.post(
        "/api/v1/query",
        json=sample_query_payload,
    )

    assert response.status_code == 200

    body = response.json()
    assert body["answer"]
    assert body["needs_vet_followup"] is False
    assert body["confidence"] == "high"
    assert body["retrieval_count"] == 1
    assert len(body["sources"]) == 1
    assert body["sources"][0]["document_id"] == "doc-demo-1"
    assert body["sources"][0]["category"] == "nutrition"
    assert body["used_filters"]["species"] == "dog"
    assert body["used_filters"]["life_stage"] == "adult"
    assert "generated_at" in body
    assert "review_draft" in body
    assert body["review_draft"] is None
    assert body["answer_source"] == "internal"
    assert body["knowledge_status"] == "approved"


def test_query_endpoint_always_strips_review_draft_for_public_api(sample_query_payload):
    """Even if the orchestrator returned review text, /query must not expose it."""
    leaky = _LeakyReviewDraftOrchestrator()
    app.dependency_overrides[get_orchestrator] = lambda: leaky
    app.dependency_overrides[get_request_auth_for_query] = conftest_mod._override_query_auth
    try:
        with TestClient(app) as test_client:
            response = test_client.post("/api/v1/query", json=sample_query_payload)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["review_draft"] is None
    assert body["answer_source"] == "internal"
    assert body["knowledge_status"] == "approved"


@pytest.mark.no_auth_override
def test_query_endpoint_returns_401_when_bearer_token_is_missing(client, sample_query_payload):
    response = client.post(
        "/api/v1/query",
        json=sample_query_payload,
    )

    assert response.status_code == 401
    body = response.json()
    assert "Missing bearer access token" in body["detail"]


@pytest.mark.no_auth_override
def test_query_endpoint_returns_401_when_access_token_is_invalid(
    client, sample_query_payload, monkeypatch
):
    def _invalid(_token: str):
        raise HTTPException(status_code=401, detail="Invalid access token.")

    monkeypatch.setattr("src.security.auth_jwt.validate_access_token", _invalid)

    response = client.post(
        "/api/v1/query",
        json=sample_query_payload,
        headers={"Authorization": "Bearer invalid-token"},
    )

    assert response.status_code == 401
    body = response.json()
    assert body["detail"] == "Invalid access token."


@pytest.mark.no_auth_override
def test_query_endpoint_returns_403_when_missing_query_permission(
    client, sample_query_payload, monkeypatch
):
    def _validate(_token: str):
        return {"sub": "auth0|no-query", "permissions": ["ingest:write"]}

    monkeypatch.setattr("src.security.auth_jwt.validate_access_token", _validate)

    response = client.post(
        "/api/v1/query",
        json=sample_query_payload,
        headers={"Authorization": "Bearer any-token"},
    )

    assert response.status_code == 403
    body = response.json()
    assert "query:ask" in body["detail"]


def test_query_endpoint_returns_422_for_invalid_payload(client):
    payload = {
        "question": "hey",
    }

    response = client.post(
        "/api/v1/query",
        json=payload,
    )

    assert response.status_code == 422
