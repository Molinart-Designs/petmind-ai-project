"""Contrato público de ``QueryResponse`` (serialización segura)."""

from datetime import datetime, timezone

from src.api.schemas import QueryResponse, query_response_dict_from_orchestrator


def test_query_response_dict_from_orchestrator_strips_internal_only_keys() -> None:
    """Orchestrator may attach logging-only keys; public contract must ignore them."""
    raw = {
        "answer": "ok",
        "review_draft": None,
        "needs_vet_followup": False,
        "confidence": "high",
        "sources": [],
        "retrieval_count": 0,
        "used_filters": {},
        "disclaimers": [],
        "generated_at": datetime.now(timezone.utc),
        "answer_source": "internal",
        "knowledge_status": "approved",
        "used_external": True,
        "internal_debug_foo": "bar",
    }
    public = query_response_dict_from_orchestrator(raw)
    assert "used_external" not in public
    assert "internal_debug_foo" not in public
    r = QueryResponse.model_validate({**public, "review_draft": None})
    assert r.answer == "ok"


def test_query_response_json_always_serializes_review_draft_as_null() -> None:
    """Defensa en esquema: aunque se construyera con texto interno, el dump JSON no lo expone."""
    r = QueryResponse(
        answer="Visible al usuario.",
        review_draft="CONTENIDO_INTERNO_NO_DEBE_SALIR",
        needs_vet_followup=False,
        confidence="high",
        generated_at=datetime.now(timezone.utc),
    )
    payload = r.model_dump(mode="json")
    assert payload["review_draft"] is None
