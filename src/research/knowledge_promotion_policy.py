"""
Explicit knowledge promotion policy for external / provisional bundles.

Approved rows in ``research_candidates`` are never created automatically from AI-generated
synthesis alone. External fallback persistence defaults to ``provisional``; medical or
heuristic-sensitive topics start as ``needs_review`` until a curator manually promotes.

Status transitions exposed here are the canonical gates for application code; ingest and
refresh workers should rely on these helpers instead of inventing ad-hoc status strings.
"""

from __future__ import annotations

import re
from typing import Literal

from src.research.schemas import KnowledgeRecordStatus

ContentSensitivity = Literal["general", "medical", "behavioral", "nutrition", "other"]
PromotionSource = Literal["manual_curator", "ai_synthesis"]

_MEDICAL_TOPIC_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bdiagnos",
        r"\bmedicat",
        r"\bprescri",
        r"\btumor",
        r"\bcancer",
        r"\bsurger",
        r"\bparvo",
        r"\bseizure",
        r"\binsulin",
        r"\bemergency\s+vet",
        r"\bchronic\s+kidney",
        r"\brenal\s+failure",
    )
)


class KnowledgePromotionError(ValueError):
    """Raised when a requested lifecycle transition violates the promotion policy."""


def initial_status_for_external_fallback(
    content_sensitivity: ContentSensitivity,
    *,
    topic: str | None,
    species: str | None,
    life_stage: str | None,
    category: str | None,
) -> tuple[str, bool]:
    """
    Initial DB status for a newly persisted external research bundle.

    Never returns ``approved``. Defaults to ``provisional``; medical sensitivity or
    heuristic topic hits yield ``needs_review`` (no auto-promotion to approved knowledge).

    Returns ``(status_value, auto_promotion_blocked)`` where ``auto_promotion_blocked`` is
    True when the row is on the sensitive / medical lane.
    """
    texts = " ".join(t for t in (topic, species, life_stage, category) if t)
    medical_blob = texts.lower()
    heuristic = any(p.search(medical_blob) for p in _MEDICAL_TOPIC_PATTERNS)
    if content_sensitivity == "medical" or heuristic:
        return KnowledgeRecordStatus.needs_review.value, True
    return KnowledgeRecordStatus.provisional.value, False


def assert_initial_external_status_never_approved(status: str) -> None:
    """Runtime guard: ingest paths must not assign ``approved`` to fresh external bundles."""
    if status == KnowledgeRecordStatus.approved.value:
        raise KnowledgePromotionError(
            "Invariant violated: external fallback bundles must not be persisted as approved"
        )


def is_medical_or_heuristic_gated_lane(
    content_sensitivity: ContentSensitivity,
    *,
    topic: str | None,
    species: str | None,
    life_stage: str | None,
    category: str | None,
) -> bool:
    """
    True si el bundle cae en vía **needs_review** (``content_sensitivity=medical`` o heurística de tema).

    Esa vía **no** admite ``approved`` en ingest ni promoción automática por refresh; solo
    transiciones explícitas con ``promotion_source=manual_curator``.
    """
    _, blocked = initial_status_for_external_fallback(
        content_sensitivity,
        topic=topic,
        species=species,
        life_stage=life_stage,
        category=category,
    )
    return blocked


def assert_sensitive_medical_bundle_policies(
    *,
    ingest_initial_status: str,
    content_sensitivity: ContentSensitivity,
    topic: str | None,
    species: str | None,
    life_stage: str | None,
    category: str | None,
) -> None:
    """
    Invariantes en ingest de fallback externo (falla con :class:`KnowledgePromotionError`):

    - Nunca ``approved`` en la primera escritura.
    - Carril médico/heurístico → estado inicial **needs_review** (no ``provisional``).
    """
    assert_initial_external_status_never_approved(ingest_initial_status)
    if is_medical_or_heuristic_gated_lane(
        content_sensitivity,
        topic=topic,
        species=species,
        life_stage=life_stage,
        category=category,
    ):
        if ingest_initial_status != KnowledgeRecordStatus.needs_review.value:
            raise KnowledgePromotionError(
                "Medical or heuristic-gated external bundles must be persisted as needs_review."
            )


def assert_refresh_status_never_auto_approved(new_status: str) -> None:
    """Runtime guard: refresh / TTL workers must never assign ``approved`` automatically."""
    if new_status == KnowledgeRecordStatus.approved.value:
        raise KnowledgePromotionError(
            "Invariant violated: refresh workers must not auto-promote external bundles to approved."
        )


def transition_provisional_to_approved(
    current_status: str,
    *,
    promotion_source: PromotionSource,
) -> str:
    """
    provisional → approved (human gate only).

    ``promotion_source`` must be ``manual_curator``; AI synthesis alone cannot approve.
    """
    if current_status != KnowledgeRecordStatus.provisional.value:
        raise KnowledgePromotionError(
            f"provisional→approved requires current status 'provisional', got {current_status!r}"
        )
    if promotion_source == "ai_synthesis":
        raise KnowledgePromotionError(
            "Approved knowledge must never be created automatically from AI-generated synthesis alone."
        )
    return KnowledgeRecordStatus.approved.value


def transition_provisional_to_needs_review(current_status: str) -> str:
    """provisional → needs_review (escalation, e.g. policy change or audit flag)."""
    if current_status != KnowledgeRecordStatus.provisional.value:
        raise KnowledgePromotionError(
            f"provisional→needs_review requires current status 'provisional', got {current_status!r}"
        )
    return KnowledgeRecordStatus.needs_review.value


def transition_provisional_to_expired(current_status: str) -> str:
    """provisional → expired (TTL, failed refresh, or explicit archival)."""
    if current_status != KnowledgeRecordStatus.provisional.value:
        raise KnowledgePromotionError(
            f"provisional→expired requires current status 'provisional', got {current_status!r}"
        )
    return KnowledgeRecordStatus.expired.value


def transition_needs_review_to_approved(
    current_status: str,
    *,
    promotion_source: PromotionSource,
) -> str:
    """
    needs_review → approved (e.g. sensitive medical row after human curation).

    Same rule as provisional→approved: never from ``ai_synthesis`` alone.
    """
    if current_status != KnowledgeRecordStatus.needs_review.value:
        raise KnowledgePromotionError(
            f"needs_review→approved requires current status 'needs_review', got {current_status!r}"
        )
    if promotion_source == "ai_synthesis":
        raise KnowledgePromotionError(
            "Approved knowledge must never be created automatically from AI-generated synthesis alone."
        )
    return KnowledgeRecordStatus.approved.value
