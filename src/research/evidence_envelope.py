"""
Helpers for persisted external-research envelopes: normalized evidence rows vs synthesis blobs.

``research_candidates.evidence_json`` schema v2 keeps retrieval-derived facts under ``evidence``
and user/curator-facing LLM outputs under ``synthesis`` so downstream jobs never treat
narrative synthesis as attributable source evidence.
"""

from __future__ import annotations

from typing import Any

from src.research.evidence_extractor import EvidenceExtractionResult
from src.research.http_url_utils import domain_from_http_url


def build_normalized_evidence_records(ex: EvidenceExtractionResult) -> list[dict[str, Any]]:
    """
    One row per snippet with URL, domain, title, body, and authority — retrieval only.

    Rows are built from :class:`ResearchEvidence` plus citation URLs/titles from the
    frontend bundle (never from LLM answer or review narrative).
    """
    citation_by_id = {c.snippet_id: c for c in ex.frontend.citations}
    source_by_id = {s.id: s for s in ex.research_evidence.sources}
    rows: list[dict[str, Any]] = []
    for sn in ex.research_evidence.snippets:
        cit = citation_by_id.get(sn.id)
        src = source_by_id.get(sn.external_source_id)
        source_url = str(cit.source_url) if cit else (str(src.base_url) if src else "")
        source_title = cit.source_title if cit else None
        source_key = src.source_key if src else ""
        rows.append(
            {
                "snippet_id": sn.id,
                "snippet": sn.text,
                "source_url": source_url,
                "source_domain": domain_from_http_url(source_url),
                "source_title": source_title,
                "authority_score": float(sn.authority_score),
                "external_source_id": sn.external_source_id,
                "source_key": source_key,
            }
        )
    return rows
