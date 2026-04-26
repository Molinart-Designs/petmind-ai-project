from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from src.research.snippet_heuristics import (
    highly_relevant_direct_answer_chunk,
    normalize_article_url_for_dedupe,
)


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

INTERNAL_CONTEXT_LIMITED_DISCLAIMER = (
    "Internally approved knowledge base passages were limited for this query; the answer may still use "
    "additional provisional context from allowlisted web sources when those were supplied."
)

PROVISIONAL_EXTERNAL_PRIMARY_DISCLAIMER = (
    "No closely matching internally approved passages were retrieved; this answer relies primarily on "
    "provisional allowlisted web sources, not on the curated internal knowledge base alone."
)

SOFT_NO_CONTEXT_FALLBACK_ANSWER = (
    "I don't have enough reliable, grounded information in PetMind's knowledge base right now to answer that "
    "with confidence. Try rephrasing your question, adding a bit more detail (for example species, age, or topic), "
    "or consulting well-established pet-care resources. If your situation involves illness, injury, pain, or "
    "anything that feels urgent, seek help from an appropriate professional."
)

CONSERVATIVE_NO_CONTEXT_FALLBACK_ANSWER = (
    "I do not have enough grounded information in the current knowledge base to answer that reliably. "
    "Please consult a licensed veterinarian for case-specific advice, especially if your pet has symptoms, "
    "is in distress, or the situation may be urgent."
)

SENSITIVE_CASE_DISCLAIMER = (
    "This may require prompt evaluation by a licensed veterinarian, especially if symptoms are severe, "
    "sudden, or worsening."
)

EXTERNAL_PROVISIONAL_CONTEXT_DISCLAIMER = (
    "Part of the context comes from provisional allowlisted web sources that are not yet part of the "
    "internally approved knowledge base; prefer curated internal passages when both apply."
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
                "breed": chunk.get("breed"),
                "life_stage": chunk.get("life_stage"),
                "similarity_score": chunk.get("similarity_score"),
                "snippet": _build_snippet_from_chunk(chunk),
                "metadata": metadata,
            }
        )

    return sanitized


def _public_source_article_key(row: dict[str, Any]) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    url = (meta.get("source_url") or "").strip()
    if url:
        return normalize_article_url_for_dedupe(url) or url
    did = row.get("document_id")
    return f"internal:{did}"


def cap_sources_per_article(
    sources: list[dict[str, Any]],
    *,
    max_per_article: int = 2,
) -> list[dict[str, Any]]:
    """Keep only the top-scoring ``max_per_article`` rows per article URL (or per internal chunk)."""
    if max_per_article <= 0:
        return []
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in sources:
        buckets[_public_source_article_key(s)].append(s)
    out: list[dict[str, Any]] = []
    for group in buckets.values():
        group.sort(key=lambda r: float(r.get("similarity_score") or 0.0), reverse=True)
        out.extend(group[:max_per_article])
    return out


def _count_non_anecdotal_external_article_sources(chunks: list[dict[str, Any]]) -> int:
    urls: set[str] = set()
    for c in chunks:
        if not _is_provisional_external_chunk(c):
            continue
        meta = c.get("metadata") if isinstance(c.get("metadata"), dict) else {}
        if meta.get("anecdotal_supplemental") is True:
            continue
        u = (meta.get("source_url") or "").strip()
        if u:
            key = normalize_article_url_for_dedupe(u)
            if key:
                urls.add(key)
    return len(urls)


def _external_has_highly_relevant_direct_answer(
    chunks: list[dict[str, Any]],
    question: str,
) -> bool:
    for c in chunks:
        if not _is_provisional_external_chunk(c):
            continue
        meta = c.get("metadata") if isinstance(c.get("metadata"), dict) else {}
        if meta.get("anecdotal_supplemental") is True:
            continue
        text = (c.get("snippet") or c.get("content") or c.get("text") or "").strip()
        sc = float(c.get("similarity_score") or 0.0)
        if highly_relevant_direct_answer_chunk(
            snippet_text=text,
            similarity_score=sc,
            question=question,
        ):
            return True
    return False


def _calibrate_confidence_for_external_evidence(
    *,
    used_provisional_external: bool,
    risk_sensitive: bool,
    risk_medical: bool,
    final_confidence: str,
    retrieved_chunks: list[dict[str, Any]],
    question: str,
) -> str:
    """
    Upgrade ``low`` → ``medium`` for solid non-medical / non-sensitive external bundles.

    Weak single-source or noisy low scores stay ``low`` (no upgrade).
    """
    if not used_provisional_external or risk_sensitive or risk_medical:
        return final_confidence
    ext_signal = _external_chunks_confidence_signal(retrieved_chunks)
    n_articles = _count_non_anecdotal_external_article_sources(retrieved_chunks)
    has_direct = _external_has_highly_relevant_direct_answer(retrieved_chunks, question)
    two_source_ok = n_articles >= 2 and ext_signal is not None and ext_signal >= 0.45
    if final_confidence != "low":
        return final_confidence
    if has_direct:
        return "medium"
    if two_source_ok:
        return "medium"
    return final_confidence


def _is_provisional_external_chunk(chunk: dict[str, Any]) -> bool:
    meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    return meta.get("provisional") is True or meta.get("layer") == "trusted_external"


def _is_reddit_supplemental_source_row(row: dict[str, Any]) -> bool:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if meta.get("anecdotal_supplemental") is True:
        return True
    dom = (meta.get("source_domain") or "").lower()
    return "reddit.com" in dom


def prioritize_sources_for_public_response(
    sources: list[dict[str, Any]],
    *,
    max_items: int = 8,
    max_anecdotal_tier: int = 1,
) -> list[dict[str, Any]]:
    """
    Higher ``similarity_score`` first.

    Reddit rows are omitted entirely unless at least one external **trusted_evidence_anchor** row
    is present (same rule as L2 selection / persistence).
    """
    has_external_anchor = any(
        isinstance(s.get("metadata"), dict)
        and s["metadata"].get("layer") == "trusted_external"
        and s["metadata"].get("trusted_evidence_anchor") is True
        for s in sources
    )
    ranked = sorted(
        sources,
        key=lambda s: float(s.get("similarity_score") or 0.0),
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    weak_used = 0
    for row in ranked:
        if len(out) >= max_items:
            break
        if _is_reddit_supplemental_source_row(row):
            if not has_external_anchor:
                continue
            if weak_used >= max_anecdotal_tier:
                continue
            weak_used += 1
        out.append(row)
    return out


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
    *,
    provisional_external_merged: bool = False,
) -> dict[str, Any]:
    """
    ``retrieved_chunks`` here is the **internal-only** slice used to judge KB strength.

    When ``provisional_external_merged`` is True, weak internal grounding must **not** imply
    “no knowledge base context” in the user-facing sense — external provisional evidence may still apply.
    """
    if not retrieved_chunks:
        disclaimers = [DEFAULT_DISCLAIMER]
        if provisional_external_merged:
            disclaimers.append(PROVISIONAL_EXTERNAL_PRIMARY_DISCLAIMER)
        else:
            disclaimers.append(NO_CONTEXT_DISCLAIMER)
        return {
            "has_any_context": False,
            "has_sufficient_context": False,
            "retrieval_count": 0,
            "matched_count": 0,
            "top_score": None,
            "disclaimers": disclaimers,
        }

    scores = [
        chunk.get("similarity_score")
        for chunk in retrieved_chunks
        if isinstance(chunk.get("similarity_score"), (int, float))
    ]

    if not scores:
        disclaimers = [DEFAULT_DISCLAIMER]
        if provisional_external_merged:
            disclaimers.append(INTERNAL_CONTEXT_LIMITED_DISCLAIMER)
        else:
            disclaimers.append(LOW_CONTEXT_DISCLAIMER)
        return {
            "has_any_context": True,
            "has_sufficient_context": False,
            "retrieval_count": len(retrieved_chunks),
            "matched_count": 0,
            "top_score": None,
            "disclaimers": disclaimers,
        }

    matched_scores = [score for score in scores if score >= similarity_threshold]
    top_score = max(scores)

    has_any_context = len(retrieved_chunks) > 0
    has_sufficient_context = len(matched_scores) > 0

    disclaimers = [DEFAULT_DISCLAIMER]
    if not has_sufficient_context:
        if provisional_external_merged:
            disclaimers.append(INTERNAL_CONTEXT_LIMITED_DISCLAIMER)
        else:
            disclaimers.append(LOW_CONTEXT_DISCLAIMER)

    return {
        "has_any_context": has_any_context,
        "has_sufficient_context": has_sufficient_context,
        "retrieval_count": len(retrieved_chunks),
        "matched_count": len(matched_scores),
        "top_score": top_score,
        "disclaimers": disclaimers,
    }


def build_safe_fallback_answer(
    question: str,
    retrieved_chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Safe answer when there is no LLM context (no internal chunks and no usable external merge).

    Non-medical, non-sensitive questions get a softer message that does **not** steer users to a vet by default.
    Medical or sensitive questions keep the explicit veterinary escalation path.
    """
    retrieved_chunks = retrieved_chunks or []
    risk = assess_query_risk(question)

    if risk.is_medical:
        answer = CONSERVATIVE_NO_CONTEXT_FALLBACK_ANSWER
        needs_vet_followup = True
        disclaimers = sorted(set(risk.disclaimers + [NO_CONTEXT_DISCLAIMER]))
    else:
        answer = SOFT_NO_CONTEXT_FALLBACK_ANSWER
        needs_vet_followup = False
        disclaimers = sorted(set(risk.disclaimers + [NO_CONTEXT_DISCLAIMER]))

    return {
        "answer": answer,
        "review_draft": None,
        "needs_vet_followup": needs_vet_followup,
        "confidence": "low",
        "sources": _sanitize_sources(retrieved_chunks),
        "retrieval_count": len(retrieved_chunks),
        "used_filters": {},
        "disclaimers": disclaimers,
        "answer_source": "fallback",
        "knowledge_status": "none",
    }


def _score_to_confidence(top_score: float | None) -> str:
    if top_score is None:
        return "low"
    if top_score >= 0.80:
        return "high"
    if top_score >= 0.60:
        return "medium"
    return "low"


def _external_chunks_confidence_signal(chunks: list[dict[str, Any]]) -> float | None:
    scores = [
        float(c["similarity_score"])
        for c in chunks
        if _is_provisional_external_chunk(c) and isinstance(c.get("similarity_score"), (int, float))
    ]
    if not scores:
        return None
    return max(scores)


def postprocess_answer(
    answer: str,
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    similarity_threshold: float,
    *,
    chunks_for_grounding: list[dict[str, Any]] | None = None,
    used_provisional_external: bool = False,
) -> dict[str, Any]:
    risk = assess_query_risk(question)
    grounding_chunks = retrieved_chunks if chunks_for_grounding is None else chunks_for_grounding
    grounding = assess_retrieval_grounding(
        grounding_chunks,
        similarity_threshold,
        provisional_external_merged=used_provisional_external,
    )

    grounding_confidence = _score_to_confidence(grounding["top_score"])
    final_confidence = grounding_confidence

    ext_signal = _external_chunks_confidence_signal(retrieved_chunks) if used_provisional_external else None
    if used_provisional_external and ext_signal is not None:
        ext_conf = _score_to_confidence(ext_signal)
        order = {"low": 0, "medium": 1, "high": 2}
        if order[ext_conf] > order[final_confidence]:
            final_confidence = ext_conf

    if risk.is_sensitive:
        final_confidence = "low"
    elif risk.is_medical and final_confidence == "high":
        final_confidence = "medium"
    elif risk.is_medical and used_provisional_external and final_confidence == "low" and ext_signal is not None:
        if ext_signal >= 0.52:
            final_confidence = "medium"

    final_confidence = _calibrate_confidence_for_external_evidence(
        used_provisional_external=used_provisional_external,
        risk_sensitive=risk.is_sensitive,
        risk_medical=risk.is_medical,
        final_confidence=final_confidence,
        retrieved_chunks=retrieved_chunks,
        question=question,
    )

    final_disclaimers = sorted(set(risk.disclaimers + grounding["disclaimers"]))

    if risk.needs_vet_followup:
        final_answer = (
            f"{answer.strip()}\n\nPlease seek veterinary guidance if symptoms are severe, sudden, worsening, "
            "or if you are concerned about your pet's safety."
        )
    else:
        final_answer = answer.strip()

    raw_sources = _sanitize_sources(retrieved_chunks)
    raw_sources = cap_sources_per_article(raw_sources, max_per_article=2)
    sources = prioritize_sources_for_public_response(raw_sources, max_items=8, max_anecdotal_tier=1)

    return {
        "answer": final_answer,
        "review_draft": None,
        "needs_vet_followup": risk.needs_vet_followup,
        "confidence": final_confidence,
        "sources": sources,
        "retrieval_count": len(retrieved_chunks),
        "used_filters": {},
        "disclaimers": final_disclaimers,
        "answer_source": "internal",
        "knowledge_status": "approved",
    }