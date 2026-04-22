from dataclasses import dataclass
from typing import Any


SENSITIVE_KEYWORDS = {
    "emergency",
    "urgent",
    "seizure",
    "seizures",
    "collapsed",
    "collapse",
    "unconscious",
    "not breathing",
    "difficulty breathing",
    "trouble breathing",
    "blood",
    "bleeding",
    "vomiting blood",
    "bloody diarrhea",
    "poison",
    "poisoning",
    "toxicity",
    "toxic",
    "can't stand",
    "cannot stand",
    "severe pain",
    "extreme pain",
    "distended abdomen",
    "bloat",
    "hit by car",
    "broken bone",
    "fracture",
}

MEDICAL_KEYWORDS = {
    "diagnosis",
    "diagnose",
    "disease",
    "illness",
    "infection",
    "fever",
    "pain",
    "vomiting",
    "diarrhea",
    "lethargy",
    "cough",
    "limping",
    "wound",
    "medication",
    "dose",
    "dosage",
    "treatment",
    "prescription",
    "symptom",
    "symptoms",
}

DEFAULT_DISCLAIMER = (
    "PetMind AI provides educational guidance based on curated information and does not replace "
    "professional veterinary evaluation."
)

LOW_CONTEXT_DISCLAIMER = (
    "The retrieved context is limited, so the answer may be incomplete or lower confidence."
)

NO_CONTEXT_DISCLAIMER = (
    "There is not enough grounded context in the knowledge base to provide a reliable answer."
)

SENSITIVE_CASE_DISCLAIMER = (
    "This may require prompt evaluation by a licensed veterinarian, especially if symptoms are severe, "
    "sudden, or worsening."
)


@dataclass
class GuardrailDecision:
    is_sensitive: bool
    is_medical: bool
    needs_vet_followup: bool
    allow_answer: bool
    confidence: str
    reasons: list[str]
    disclaimers: list[str]


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _contains_any(text: str, keywords: set[str]) -> list[str]:
    matches: list[str] = []
    for keyword in keywords:
        if keyword in text:
            matches.append(keyword)
    return sorted(set(matches))


def _build_snippet_from_chunk(chunk: dict[str, Any], max_length: int = 280) -> str | None:
    snippet = chunk.get("snippet")
    if snippet:
        return snippet

    raw_text = (chunk.get("content") or chunk.get("text") or "").strip()
    if not raw_text:
        return None

    normalized = " ".join(raw_text.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."


def _sanitize_sources(retrieved_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []

    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        sanitized.append(
            {
                "document_id": chunk.get("document_id"),
                "chunk_id": chunk.get("chunk_id"),
                "title": chunk.get("title"),
                "source": chunk.get("source"),
                "category": chunk.get("category"),
                "species": chunk.get("species"),
                "life_stage": chunk.get("life_stage"),
                "similarity_score": chunk.get("similarity_score"),
                "snippet": _build_snippet_from_chunk(chunk),
                "metadata": metadata,
            }
        )

    return sanitized


def assess_query_risk(question: str) -> GuardrailDecision:
    normalized_question = _normalize_text(question)

    sensitive_matches = _contains_any(normalized_question, SENSITIVE_KEYWORDS)
    medical_matches = _contains_any(normalized_question, MEDICAL_KEYWORDS)

    is_sensitive = len(sensitive_matches) > 0
    is_medical = is_sensitive or len(medical_matches) > 0
    needs_vet_followup = (
        is_sensitive
        or "diagnosis" in normalized_question
        or "dose" in normalized_question
        or "dosage" in normalized_question
    )

    reasons: list[str] = []
    disclaimers = [DEFAULT_DISCLAIMER]

    if sensitive_matches:
        reasons.append(f"Sensitive terms detected: {', '.join(sensitive_matches)}")
        disclaimers.append(SENSITIVE_CASE_DISCLAIMER)

    if medical_matches and not sensitive_matches:
        reasons.append(f"Medical terms detected: {', '.join(medical_matches)}")

    confidence = "low" if is_sensitive else "medium" if is_medical else "high"

    return GuardrailDecision(
        is_sensitive=is_sensitive,
        is_medical=is_medical,
        needs_vet_followup=needs_vet_followup,
        allow_answer=True,
        confidence=confidence,
        reasons=reasons,
        disclaimers=disclaimers,
    )


def assess_retrieval_grounding(
    retrieved_chunks: list[dict[str, Any]],
    similarity_threshold: float,
) -> dict[str, Any]:
    if not retrieved_chunks:
        return {
            "has_any_context": False,
            "has_sufficient_context": False,
            "matched_count": 0,
            "top_score": None,
            "disclaimers": [DEFAULT_DISCLAIMER, NO_CONTEXT_DISCLAIMER],
        }

    scores = [
        chunk.get("similarity_score")
        for chunk in retrieved_chunks
        if isinstance(chunk.get("similarity_score"), (int, float))
    ]

    if not scores:
        return {
            "has_any_context": True,
            "has_sufficient_context": False,
            "matched_count": 0,
            "top_score": None,
            "disclaimers": [DEFAULT_DISCLAIMER, LOW_CONTEXT_DISCLAIMER],
        }

    matched_scores = [score for score in scores if score >= similarity_threshold]
    top_score = max(scores)

    has_any_context = len(retrieved_chunks) > 0
    has_sufficient_context = len(matched_scores) > 0

    disclaimers = [DEFAULT_DISCLAIMER]
    if not has_sufficient_context:
        disclaimers.append(LOW_CONTEXT_DISCLAIMER)

    return {
        "has_any_context": has_any_context,
        "has_sufficient_context": has_sufficient_context,
        "matched_count": len(matched_scores),
        "top_score": top_score,
        "disclaimers": disclaimers,
    }


def build_safe_fallback_answer(
    question: str,
    retrieved_chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    retrieved_chunks = retrieved_chunks or []
    risk = assess_query_risk(question)

    answer = (
        "I do not have enough grounded information in the current knowledge base to answer that reliably. "
        "Please consult a licensed veterinarian for case-specific advice, especially if your pet has symptoms, "
        "is in distress, or the situation may be urgent."
    )

    return {
        "answer": answer,
        "needs_vet_followup": True if risk.is_medical else risk.needs_vet_followup,
        "confidence": "low",
        "sources": _sanitize_sources(retrieved_chunks),
        "retrieval_count": len(retrieved_chunks),
        "used_filters": {},
        "disclaimers": sorted(set(risk.disclaimers + [NO_CONTEXT_DISCLAIMER])),
    }


def _score_to_confidence(top_score: float | None) -> str:
    if top_score is None:
        return "low"
    if top_score >= 0.80:
        return "high"
    if top_score >= 0.60:
        return "medium"
    return "low"


def postprocess_answer(
    answer: str,
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    similarity_threshold: float,
) -> dict[str, Any]:
    risk = assess_query_risk(question)
    grounding = assess_retrieval_grounding(retrieved_chunks, similarity_threshold)

    grounding_confidence = _score_to_confidence(grounding["top_score"])
    final_confidence = grounding_confidence

    if risk.is_sensitive:
        final_confidence = "low"
    elif risk.is_medical and final_confidence == "high":
        final_confidence = "medium"

    final_disclaimers = sorted(set(risk.disclaimers + grounding["disclaimers"]))

    if risk.needs_vet_followup:
        final_answer = (
            f"{answer.strip()}\n\nPlease seek veterinary guidance if symptoms are severe, sudden, worsening, "
            "or if you are concerned about your pet's safety."
        )
    else:
        final_answer = answer.strip()

    return {
        "answer": final_answer,
        "needs_vet_followup": risk.needs_vet_followup,
        "confidence": final_confidence,
        "sources": _sanitize_sources(retrieved_chunks),
        "retrieval_count": len(retrieved_chunks),
        "used_filters": {},
        "disclaimers": final_disclaimers,
    }