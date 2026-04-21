from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.core.config import settings
from src.security.auth import require_api_key


def create_test_app() -> FastAPI:
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(require_api_key)])
    async def protected_route():
        return {"ok": True}

    return app


def test_require_api_key_allows_request_when_header_is_valid():
    app = create_test_app()
    client = TestClient(app)

    response = client.get(
        "/protected",
        headers={settings.api_key_header_name: settings.api_key},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_require_api_key_returns_401_when_header_is_missing():
    app = create_test_app()
    client = TestClient(app)

    response = client.get("/protected")

    assert response.status_code == 401
    assert response.json() == {
        "detail": f"Missing API key. Expected header: {settings.api_key_header_name}"
    }


def test_require_api_key_returns_401_when_header_is_invalid():
    app = create_test_app()
    client = TestClient(app)

    response = client.get(
        "/protected",
        headers={settings.api_key_header_name: "invalid-key"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid API key."}