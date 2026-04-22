from src.security.guardrails import (
    DEFAULT_DISCLAIMER,
    LOW_CONTEXT_DISCLAIMER,
    NO_CONTEXT_DISCLAIMER,
    SENSITIVE_CASE_DISCLAIMER,
    assess_query_risk,
    assess_retrieval_grounding,
    build_safe_fallback_answer,
    postprocess_answer,
)


def test_assess_query_risk_detects_sensitive_question():
    question = "My dog is having seizures and difficulty breathing, what should I do?"

    result = assess_query_risk(question)

    assert result.is_sensitive is True
    assert result.is_medical is True
    assert result.needs_vet_followup is True
    assert result.allow_answer is True
    assert result.confidence == "low"
    assert DEFAULT_DISCLAIMER in result.disclaimers
    assert SENSITIVE_CASE_DISCLAIMER in result.disclaimers
    assert len(result.reasons) > 0


def test_assess_query_risk_detects_medical_but_not_sensitive_question():
    question = "My dog has mild diarrhea, what should I monitor?"

    result = assess_query_risk(question)

    assert result.is_sensitive is False
    assert result.is_medical is True
    assert result.confidence == "medium"
    assert DEFAULT_DISCLAIMER in result.disclaimers


def test_assess_retrieval_grounding_returns_false_when_no_chunks():
    result = assess_retrieval_grounding(
        retrieved_chunks=[],
        similarity_threshold=0.75,
    )

    assert result["has_any_context"] is False
    assert result["has_sufficient_context"] is False
    assert result["matched_count"] == 0
    assert result["top_score"] is None
    assert DEFAULT_DISCLAIMER in result["disclaimers"]
    assert NO_CONTEXT_DISCLAIMER in result["disclaimers"]


def test_assess_retrieval_grounding_returns_true_when_chunk_meets_threshold():
    retrieved_chunks = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "similarity_score": 0.82,
            "snippet": "Adult dogs need consistent hydration and regular feeding.",
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-1",
            "similarity_score": 0.68,
            "snippet": "General pet care tips.",
        },
    ]

    result = assess_retrieval_grounding(
        retrieved_chunks=retrieved_chunks,
        similarity_threshold=0.75,
    )

    assert result["has_any_context"] is True
    assert result["has_sufficient_context"] is True
    assert result["matched_count"] == 1
    assert result["top_score"] == 0.82
    assert LOW_CONTEXT_DISCLAIMER not in result["disclaimers"]


def test_build_safe_fallback_answer_returns_low_confidence_and_vet_guidance():
    question = "My dog is vomiting blood, what should I do?"
    retrieved_chunks = []

    result = build_safe_fallback_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    assert result["confidence"] == "low"
    assert result["needs_vet_followup"] is True
    assert result["retrieval_count"] == 0
    assert result["sources"] == []
    assert NO_CONTEXT_DISCLAIMER in result["disclaimers"]
    assert "licensed veterinarian" in result["answer"]


def test_postprocess_answer_appends_vet_guidance_for_sensitive_question():
    answer = "Based on the available context, breathing difficulty can be serious."
    question = "My dog has trouble breathing and collapsed briefly."
    retrieved_chunks = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "similarity_score": 0.91,
            "snippet": "Breathing difficulty can indicate an urgent issue in pets.",
        }
    ]

    result = postprocess_answer(
        answer=answer,
        question=question,
        retrieved_chunks=retrieved_chunks,
        similarity_threshold=0.75,
    )

    assert result["confidence"] == "low"
    assert result["needs_vet_followup"] is True
    assert result["retrieval_count"] == 1
    assert "Please seek veterinary guidance" in result["answer"]
    assert DEFAULT_DISCLAIMER in result["disclaimers"]


def test_postprocess_answer_returns_low_confidence_when_grounding_is_insufficient():
    answer = "You may monitor hydration and appetite."
    question = "How can I help my adult dog recover from mild stomach upset?"
    retrieved_chunks = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "similarity_score": 0.51,
            "snippet": "General pet wellness guidance.",
        }
    ]

    result = postprocess_answer(
        answer=answer,
        question=question,
        retrieved_chunks=retrieved_chunks,
        similarity_threshold=0.75,
    )

    assert result["confidence"] == "low"
    assert result["needs_vet_followup"] is False
    assert LOW_CONTEXT_DISCLAIMER in result["disclaimers"]


def test_postprocess_answer_keeps_high_confidence_for_non_medical_grounded_question():
    answer = "Adult dogs generally benefit from consistent feeding schedules."
    question = "What is a good feeding routine for an adult dog?"
    retrieved_chunks = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "similarity_score": 0.89,
            "snippet": "Adult dogs benefit from routine feeding schedules.",
        }
    ]

    result = postprocess_answer(
        answer=answer,
        question=question,
        retrieved_chunks=retrieved_chunks,
        similarity_threshold=0.75,
    )

    assert result["confidence"] == "high"
    assert result["needs_vet_followup"] is False
    assert result["retrieval_count"] == 1
    assert LOW_CONTEXT_DISCLAIMER not in result["disclaimers"]