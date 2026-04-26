"""
Snippet quality and persistence eligibility for trusted external evidence (Layer 2 → 3).

Kept free of imports from ``evidence_extractor`` to avoid circular imports with ``evidence_envelope``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from src.research.domain_authority import (
    snippet_is_anchor_trusted_evidence,
    snippet_is_supplemental_reddit,
)
from src.research.http_url_utils import domain_from_http_url

if TYPE_CHECKING:
    from src.research.evidence_extractor import EvidenceExtractionResult

_PLACEHOLDER_HOST_SUFFIXES: tuple[str, ...] = (
    ".example",
    ".example.com",
    ".example.net",
    ".example.org",
    ".invalid",
    ".localhost",
    ".test",
)
_PLACEHOLDER_EXACT_HOSTS: frozenset[str] = frozenset(
    {
        "example.com",
        "example.net",
        "example.org",
        "localhost",
        "127.0.0.1",
    }
)

MIN_EVIDENCE_SNIPPET_CHARS = 40
MIN_EVIDENCE_SNIPPET_WORDS = 6
MIN_SNIPPETS_FOR_PERSISTENCE = 2


def _normalize_host(host: str) -> str:
    h = (host or "").strip().lower().rstrip(".")
    if h.startswith("www."):
        h = h[4:]
    return h


def is_placeholder_evidence_hostname(host: str) -> bool:
    """True for RFC 2606-style or loopback hosts that must not anchor persisted evidence."""
    h = _normalize_host(host)
    if not h:
        return True
    if h in _PLACEHOLDER_EXACT_HOSTS:
        return True
    return any(h.endswith(suf) for suf in _PLACEHOLDER_HOST_SUFFIXES)


def snippet_text_meets_evidence_quality(text: str) -> bool:
    """
    Reject tiny fragments, stub boilerplate, and token-sparse lines before persistence / claims.
    """
    raw = " ".join(text.replace("\r\n", "\n").split()).strip()
    if len(raw) < MIN_EVIDENCE_SNIPPET_CHARS:
        return False
    words = [w for w in re.split(r"\s+", raw) if w]
    if len(words) < MIN_EVIDENCE_SNIPPET_WORDS:
        return False
    lowered = raw.lower()
    if "stub retrieval for" in lowered and "replace stubexternalretrievalprovider" in lowered:
        return False
    alpha_tokens = sum(1 for w in words if re.search(r"[a-zA-Záéíóúñ]{2,}", w))
    if alpha_tokens < 4:
        return False
    return True


def evidence_bundle_eligible_for_persistence(ex: EvidenceExtractionResult) -> tuple[bool, list[str]]:
    """Return (eligible, rejection reasons) for research_candidate persistence and review synthesis."""
    from src.research.evidence_envelope import build_normalized_evidence_records

    reasons: list[str] = []
    snippets = ex.research_evidence.snippets
    if len(snippets) < MIN_SNIPPETS_FOR_PERSISTENCE:
        reasons.append(f"too_few_snippets_need_{MIN_SNIPPETS_FOR_PERSISTENCE}_got_{len(snippets)}")

    # reddit.com is supplemental only: never persist bundles that rely on Reddit without an anchor.
    if any(snippet_is_supplemental_reddit(sn) for sn in snippets) and not any(
        snippet_is_anchor_trusted_evidence(sn) for sn in snippets
    ):
        reasons.append("reddit_requires_anchor_trusted_snippet")

    for sn in snippets:
        if not snippet_text_meets_evidence_quality(sn.text):
            reasons.append(f"low_quality_snippet:{sn.id[:16]}")

    rows = build_normalized_evidence_records(ex)
    for row in rows:
        dom = (row.get("source_domain") or "").lower()
        url = row.get("source_url") or ""
        if not url or not dom:
            reasons.append("missing_source_url_or_domain")
            continue
        if is_placeholder_evidence_hostname(dom):
            reasons.append(f"placeholder_domain:{dom}")

    ok = len(reasons) == 0
    return ok, reasons
