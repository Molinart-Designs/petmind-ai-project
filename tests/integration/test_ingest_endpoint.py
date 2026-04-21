def test_ingest_endpoint_returns_201_with_valid_api_key_and_payload(
    client,
    auth_headers,
    sample_ingest_payload,
):
    response = client.post(
        "/api/v1/ingest",
        json=sample_ingest_payload,
        headers=auth_headers,
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


def test_ingest_endpoint_returns_401_when_api_key_is_missing(
    client,
    sample_ingest_payload,
):
    response = client.post(
        "/api/v1/ingest",
        json=sample_ingest_payload,
    )

    assert response.status_code == 401
    body = response.json()
    assert "Missing API key" in body["detail"]


def test_ingest_endpoint_returns_401_when_api_key_is_invalid(
    client,
    sample_ingest_payload,
):
    response = client.post(
        "/api/v1/ingest",
        json=sample_ingest_payload,
        headers={"X-API-Key": "invalid-key"},
    )

    assert response.status_code == 401
    body = response.json()
    assert body["detail"] == "Invalid API key."


def test_ingest_endpoint_returns_422_for_invalid_payload(
    client,
    auth_headers,
):
    payload = {
        "source": "demo_batch",
        "documents": [],
    }

    response = client.post(
        "/api/v1/ingest",
        json=payload,
        headers=auth_headers,
    )

    assert response.status_code == 422