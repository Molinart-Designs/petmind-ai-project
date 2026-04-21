from contextlib import contextmanager

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health_returns_200_and_healthy_when_database_is_available(monkeypatch):
    class FakeSession:
        def execute(self, *_args, **_kwargs):
            return 1

    @contextmanager
    def fake_get_db_session():
        yield FakeSession()

    monkeypatch.setattr("src.api.routes.get_db_session", fake_get_db_session)

    response = client.get("/api/v1/health")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "healthy"
    assert body["database"] == "healthy"
    assert body["service"] == "petmind-ai"
    assert "timestamp" in body


def test_health_returns_200_and_degraded_when_database_is_unavailable(monkeypatch):
    @contextmanager
    def fake_get_db_session():
        raise RuntimeError("database unavailable")
        yield  # pragma: no cover

    monkeypatch.setattr("src.api.routes.get_db_session", fake_get_db_session)

    response = client.get("/api/v1/health")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "degraded"
    assert body["database"] == "unhealthy"
    assert body["service"] == "petmind-ai"
    assert "timestamp" in body