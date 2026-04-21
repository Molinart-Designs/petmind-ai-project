def test_query_endpoint_returns_200_with_valid_api_key_and_payload(
    client,
    auth_headers,
    sample_query_payload,
):
    response = client.post(
        "/api/v1/query",
        json=sample_query_payload,
        headers=auth_headers,
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


def test_query_endpoint_returns_401_when_api_key_is_missing(
    client,
    sample_query_payload,
):
    response = client.post(
        "/api/v1/query",
        json=sample_query_payload,
    )

    assert response.status_code == 401
    body = response.json()
    assert "Missing API key" in body["detail"]


def test_query_endpoint_returns_401_when_api_key_is_invalid(
    client,
    sample_query_payload,
):
    response = client.post(
        "/api/v1/query",
        json=sample_query_payload,
        headers={"X-API-Key": "invalid-key"},
    )

    assert response.status_code == 401
    body = response.json()
    assert body["detail"] == "Invalid API key."


def test_query_endpoint_returns_422_for_invalid_payload(
    client,
    auth_headers,
):
    payload = {
        "question": "hey"
    }

    response = client.post(
        "/api/v1/query",
        json=payload,
        headers=auth_headers,
    )

    assert response.status_code == 422