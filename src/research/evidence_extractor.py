"""
Turn trusted retrieval hits (or legacy fetched pages) into structured evidence.

Produces separate shapes for UI citations and for internal review—never final user answers.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Literal, Protocol, runtime_checkable
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, Field, ConfigDict, HttpUrl

from src.research.domain_authority import (
    clamp_registry_authority,
    clamp_synthetic_authority,
    domain_authority_score,
    snippet_is_anchor_trusted_evidence,
    snippet_is_supplemental_reddit,
)
from src.research.evidence_quality import snippet_text_meets_evidence_quality
from src.research.excerpt_sanitize import clean_excerpt_for_evidence
from src.research.external_ranking import (
    composite_snippet_ranking,
    normalize_provider_relevance,
    token_overlap_score,
)
from src.research.snippet_heuristics import (
    breed_specificity_signal,
    dedupe_similar_claim_units,
    direct_answer_signal,
    infer_snippet_scope_fields,
    noise_signal,
    should_discard_snippet_unit,
)
from src.research.http_url_utils import domain_from_http_url
from src.research.schemas import ExtractedSnippet, ExternalSource, ExternalSourceType, ResearchEvidence
from src.research.web_retriever import FetchedPage, TrustedRetrievalResult, TrustedSourceHit
from src.utils.logger import get_logger

logger = get_logger(__name__)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace and strip ends (preserves single spaces between words)."""
    return " ".join(text.replace("\r\n", "\n").replace("\r", "\n").split())


def split_excerpt_into_claim_units(text: str, *, max_units: int = 8) -> list[str]:
    """
    Split normalized prose into short claim-oriented units (sentence-like chunks).

    Falls back to a single bounded unit when there are no clear sentence boundaries.
    """
    normalized = normalize_whitespace(text)
    if not normalized:
        return []

    parts = re.split(r"(?<=[.!?])\s+", normalized)
    units: list[str] = []
    for p in parts:
        u = normalize_whitespace(p)
        if len(u) >= 28:
            units.append(u)
    if not units:
        cap = min(len(normalized), 800)
        chunk = normalize_whitespace(normalized[:cap])
        return [chunk] if chunk else []

    return units[:max_units]


def _stable_external_source_id(source_key: str) -> str:
    """Align with ``source_registry`` synthetic ids for the same ``source_key``."""
    return str(uuid5(NAMESPACE_URL, f"petmind:trusted-source:{source_key}"))


def _stable_external_source_id_for_hit(hit: TrustedSourceHit) -> str:
    """Stable id per logical source key + page host so wildcard / multi-URL runs do not collapse sources."""
    host = domain_from_http_url(str(hit.url)) or "unknown-host"
    return str(uuid5(NAMESPACE_URL, f"petmind:trusted-source:{hit.source_key}:{host}"))


def _snippet_id(*, source_key: str, page_url: str, unit_index: int, unit_text: str) -> str:
    digest = hashlib.sha256(unit_text.encode("utf-8")).hexdigest()[:20]
    return str(uuid5(NAMESPACE_URL, f"petmind:snippet:{source_key}:{page_url}:{unit_index}:{digest}"))


def _claim_id(*, snippet_id: str, claim_index: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"petmind:claim:{snippet_id}:{claim_index}"))


def _synthetic_external_source(hit: TrustedSourceHit) -> ExternalSource:
    """Minimal ``ExternalSource`` when callers did not pass a registry-backed row."""
    u = hit.url
    scheme = str(u.scheme).lower() if u.scheme else "https"
    if scheme not in ("http", "https"):
        scheme = "https"
    host = str(u.host) if u.host else ""
    if host:
        base_url = HttpUrl(f"{scheme}://{host}/")
    else:
        base_url = u

    dom = domain_from_http_url(str(u))
    auth = clamp_synthetic_authority(dom)

    return ExternalSource(
        id=_stable_external_source_id_for_hit(hit),
        source_key=hit.source_key,
        base_url=base_url,
        authority_score=auth,
        source_type=ExternalSourceType.allowlisted_web,
        topic=None,
        species=None,
        breed=None,
        life_stage=None,
        retrieved_at=hit.retrieved_at,
        review_after=None,
    )


def _resolve_external_source(
    hit: TrustedSourceHit,
    sources_by_key: dict[str, ExternalSource] | None,
) -> ExternalSource:
    """
    Prefer registry row only when it matches the hit URL host.

    Wildcard / seed registries often register a placeholder ``base_url``; real hits must use
    per-page synthetic sources so citations and persistence keep the true document URL.
    """
    if hit.source_key == "wildcard_any_domain":
        return _synthetic_external_source(hit)
    if sources_by_key and hit.source_key in sources_by_key:
        reg = sources_by_key[hit.source_key]
        reg_host = domain_from_http_url(str(reg.base_url))
        hit_host = domain_from_http_url(str(hit.url))
        if reg_host and hit_host and reg_host == hit_host:
            auth = clamp_registry_authority(hit_host, reg.authority_score)
            if abs(auth - reg.authority_score) > 1e-9:
                return reg.model_copy(update={"authority_score": auth})
            return reg
        return _synthetic_external_source(hit)
    return _synthetic_external_source(hit)


class ClaimEvidenceRecord(BaseModel):
    """One atomic evidentiary unit tied to supporting snippet ids (no free-form answer)."""

    claim_id: str = Field(..., min_length=1, max_length=128)
    text: str = Field(..., min_length=1, max_length=8000)
    snippet_ids: list[str] = Field(default_factory=list, description="Supporting ExtractedSnippet.id values")
    source_url: str = Field(..., max_length=2000)
    source_key: str = Field(..., min_length=1, max_length=64)
    source_title: str | None = Field(default=None, max_length=500)

    model_config = ConfigDict(extra="forbid")


class FrontendEvidenceCitation(BaseModel):
    """Citation-safe row for the API layer (no review-only internals)."""

    snippet_id: str = Field(..., min_length=1, max_length=128)
    text: str = Field(..., min_length=1, max_length=8000)
    source_url: str = Field(..., max_length=2000)
    source_title: str | None = Field(default=None, max_length=500)
    source_key: str = Field(..., min_length=1, max_length=64)

    model_config = ConfigDict(extra="forbid")


class FrontendEvidenceBundle(BaseModel):
    """Evidence shaped for frontend display (provisional; not a user answer)."""

    citations: list[FrontendEvidenceCitation] = Field(default_factory=list)
    generated_at: datetime
    is_provisional: Literal[True] = True
    disclaimer: str = Field(
        default="This evidence is provisional and does not replace veterinary diagnosis or treatment.",
        max_length=500,
    )

    model_config = ConfigDict(extra="forbid")


class ReviewDraftEvidenceBundle(BaseModel):
    """Richer bundle for internal review / curation workflows."""

    snippets: list[ExtractedSnippet] = Field(default_factory=list)
    claims: list[ClaimEvidenceRecord] = Field(default_factory=list)
    sources: list[ExternalSource] = Field(default_factory=list)
    provider_id: str = Field(..., min_length=1, max_length=64)
    blocked_url_count: int = Field(default=0, ge=0)
    extraction_notes: list[str] = Field(default_factory=list)
    generated_at: datetime

    model_config = ConfigDict(extra="forbid")


class EvidenceExtractionResult(BaseModel):
    """Dual bundles plus canonical :class:`ResearchEvidence` for downstream RAG storage."""

    frontend: FrontendEvidenceBundle
    review_draft: ReviewDraftEvidenceBundle
    research_evidence: ResearchEvidence

    model_config = ConfigDict(extra="forbid")


def select_snippets_for_research_result(
    snippets: list[ExtractedSnippet],
    *,
    max_total: int,
    max_low_tier: int = 1,
) -> list[ExtractedSnippet]:
    """
    Reddit is **never** primary: without at least one higher-authority anchor snippet, all Reddit
    rows are dropped. With an anchor, at most ``max_low_tier`` Reddit snippets are kept, after all
    anchors and other non-Reddit candidates ranked by composite score.
    """
    if max_total <= 0:
        return []

    def rank_key(s: ExtractedSnippet) -> tuple[float, float]:
        return (s.ranking_score or 0.0, s.authority_score)

    anchors = [s for s in snippets if snippet_is_anchor_trusted_evidence(s)]
    redditors = [s for s in snippets if snippet_is_supplemental_reddit(s)]
    anchor_ids = {s.id for s in anchors}
    reddit_ids = {s.id for s in redditors}
    # Non-Reddit snippets that are not anchor-tier (e.g. mid-authority) still fill context before Reddit.
    rest = [s for s in snippets if s.id not in reddit_ids and s.id not in anchor_ids]

    reddit_budget = min(max_low_tier, 1, max_total) if anchors else 0

    out: list[ExtractedSnippet] = []
    for s in sorted(anchors, key=rank_key, reverse=True):
        if len(out) >= max_total:
            break
        out.append(s)
    for s in sorted(rest, key=rank_key, reverse=True):
        if len(out) >= max_total:
            break
        out.append(s)
    if reddit_budget > 0:
        for s in sorted(redditors, key=rank_key, reverse=True):
            if len(out) >= max_total or reddit_budget <= 0:
                break
            out.append(s)
            reddit_budget -= 1
    return out[:max_total]


class RetrievalEvidenceExtractor:
    """
    Build structured evidence from :class:`TrustedSourceHit` rows.

    Does not call LLMs and does not synthesize user-facing answers.
    """

    @staticmethod
    def from_trusted_hits(
        hits: Sequence[TrustedSourceHit],
        *,
        provider_id: str,
        sources_by_key: dict[str, ExternalSource] | None = None,
        blocked_url_count: int = 0,
        max_claim_units_per_hit: int = 8,
        query: str | None = None,
        max_snippets_out: int | None = None,
        max_low_tier_snippets: int = 1,
    ) -> EvidenceExtractionResult:
        now = datetime.now(timezone.utc)
        snippets: list[ExtractedSnippet] = []
        claims: list[ClaimEvidenceRecord] = []
        frontend_citations: list[FrontendEvidenceCitation] = []
        notes: list[str] = []
        source_by_id: dict[str, ExternalSource] = {}
        q = (query or "").strip()

        def _pre_rank(hit: TrustedSourceHit) -> float:
            dom = domain_from_http_url(str(hit.url))
            rel = normalize_provider_relevance(hit.relevance_score)
            auth = domain_authority_score(dom)
            return 0.55 * rel + 0.45 * auth

        for hit in sorted(list(hits), key=_pre_rank, reverse=True):
            source = _resolve_external_source(hit, sources_by_key)
            source_by_id[source.id] = source
            page_url = str(hit.url)
            excerpt_raw = normalize_whitespace(hit.excerpt)
            excerpt = clean_excerpt_for_evidence(excerpt_raw)
            if not excerpt:
                notes.append(f"skipped_empty_excerpt_after_sanitize:{hit.source_key}")
                continue

            units = split_excerpt_into_claim_units(excerpt, max_units=max_claim_units_per_hit)
            units = dedupe_similar_claim_units(units)
            if len(units) > 1:
                notes.append(f"split_into_{len(units)}_units:{page_url}")

            page_title = (hit.title or "").strip() or None
            for idx, unit in enumerate(units):
                if should_discard_snippet_unit(unit, q, page_title=page_title):
                    notes.append(f"skipped_low_value_unit:{page_url}:{idx}")
                    continue
                if not snippet_text_meets_evidence_quality(unit):
                    notes.append(f"skipped_low_quality_unit:{page_url}:{idx}")
                    continue
                sid = _snippet_id(source_key=hit.source_key, page_url=page_url, unit_index=idx, unit_text=unit)
                overlap = token_overlap_score(q, unit) if q else 0.0
                da = direct_answer_signal(unit, q) if q else 0.0
                sp = breed_specificity_signal(unit, q, page_title=page_title) if q else 0.0
                nz = noise_signal(unit, q, page_title=page_title) if q else 0.0
                rank = composite_snippet_ranking(
                    provider_relevance=hit.relevance_score,
                    source_authority=source.authority_score,
                    query_overlap=overlap,
                    direct_answer_signal=da,
                    specificity_signal=sp,
                    noise_signal=nz,
                )
                scope = infer_snippet_scope_fields(q, page_title=page_title, unit_text=unit) if q else {}
                snippet = ExtractedSnippet(
                    id=sid,
                    external_source_id=source.id,
                    text=unit,
                    evidence_page_url=page_url,
                    authority_score=source.authority_score,
                    source_type=source.source_type,
                    topic=source.topic,
                    species=scope.get("species") or source.species,
                    breed=scope.get("breed") or source.breed,
                    life_stage=scope.get("life_stage") or source.life_stage,
                    retrieved_at=hit.retrieved_at,
                    review_after=source.review_after,
                    page_title=page_title,
                    provider_relevance=hit.relevance_score,
                    ranking_score=rank,
                )
                snippets.append(snippet)
                cite_title = page_title
                cid = _claim_id(snippet_id=sid, claim_index=0)
                claims.append(
                    ClaimEvidenceRecord(
                        claim_id=cid,
                        text=unit,
                        snippet_ids=[sid],
                        source_url=page_url,
                        source_key=hit.source_key,
                        source_title=cite_title,
                    )
                )
                frontend_citations.append(
                    FrontendEvidenceCitation(
                        snippet_id=sid,
                        text=unit,
                        source_url=page_url,
                        source_title=cite_title,
                        source_key=hit.source_key,
                    )
                )

        if max_snippets_out is not None and max_snippets_out > 0 and len(snippets) > max_snippets_out:
            snippets = select_snippets_for_research_result(
                snippets,
                max_total=max_snippets_out,
                max_low_tier=max_low_tier_snippets,
            )
            sid_ok = {s.id for s in snippets}
            claims = [c for c in claims if c.snippet_ids and c.snippet_ids[0] in sid_ok]
            frontend_citations = [fc for fc in frontend_citations if fc.snippet_id in sid_ok]
            notes.append(f"capped_snippets_to_{max_snippets_out}")

        ref_ids = {s.external_source_id for s in snippets}
        sources = [s for s in source_by_id.values() if s.id in ref_ids]
        research = ResearchEvidence(snippets=snippets, sources=sources)

        frontend = FrontendEvidenceBundle(citations=frontend_citations, generated_at=now)
        review = ReviewDraftEvidenceBundle(
            snippets=snippets,
            claims=claims,
            sources=sources,
            provider_id=provider_id,
            blocked_url_count=blocked_url_count,
            extraction_notes=notes,
            generated_at=now,
        )

        logger.info(
            "Retrieval evidence extraction complete",
            extra={
                "snippet_count": len(snippets),
                "claim_count": len(claims),
                "provider_id": provider_id,
            },
        )

        return EvidenceExtractionResult(frontend=frontend, review_draft=review, research_evidence=research)

    @classmethod
    def from_retrieval_result(
        cls,
        result: TrustedRetrievalResult,
        *,
        sources_by_key: dict[str, ExternalSource] | None = None,
        max_claim_units_per_hit: int = 8,
        query: str | None = None,
        max_snippets_out: int | None = None,
        max_low_tier_snippets: int = 1,
    ) -> EvidenceExtractionResult:
        """Convenience wrapper including blocked-url counts for review drafts."""
        return cls.from_trusted_hits(
            result.hits,
            provider_id=result.provider_id,
            sources_by_key=sources_by_key,
            blocked_url_count=len(result.blocked),
            max_claim_units_per_hit=max_claim_units_per_hit,
            query=query,
            max_snippets_out=max_snippets_out,
            max_low_tier_snippets=max_low_tier_snippets,
        )

    @staticmethod
    def from_research_evidence(
        evidence: ResearchEvidence,
        *,
        provider_id: str,
        blocked_url_count: int = 0,
    ) -> EvidenceExtractionResult:
        """
        Build frontend/review bundles from an already-assembled :class:`ResearchEvidence`
        (e.g. after snippet caps), when raw hits are no longer available.
        """
        now = datetime.now(timezone.utc)
        sources_by_id = {s.id: s for s in evidence.sources}
        claims: list[ClaimEvidenceRecord] = []
        citations: list[FrontendEvidenceCitation] = []
        notes: list[str] = ["assembled_from_research_evidence"]

        for sn in evidence.snippets:
            src = sources_by_id.get(sn.external_source_id)
            page_url = (
                (sn.evidence_page_url or "").strip()
                or (str(src.base_url) if src else "")
                or "https://unknown.invalid/"
            )
            source_key = src.source_key if src else "unknown"
            source_title = (sn.page_title or "").strip() or None
            cid = _claim_id(snippet_id=sn.id, claim_index=0)
            claims.append(
                ClaimEvidenceRecord(
                    claim_id=cid,
                    text=sn.text,
                    snippet_ids=[sn.id],
                    source_url=page_url,
                    source_key=source_key,
                    source_title=source_title,
                )
            )
            citations.append(
                FrontendEvidenceCitation(
                    snippet_id=sn.id,
                    text=sn.text,
                    source_url=page_url,
                    source_title=source_title,
                    source_key=source_key,
                )
            )

        frontend = FrontendEvidenceBundle(citations=citations, generated_at=now)
        review = ReviewDraftEvidenceBundle(
            snippets=list(evidence.snippets),
            claims=claims,
            sources=list(evidence.sources),
            provider_id=provider_id,
            blocked_url_count=blocked_url_count,
            extraction_notes=notes,
            generated_at=now,
        )
        return EvidenceExtractionResult(frontend=frontend, review_draft=review, research_evidence=evidence)


def _trusted_hit_from_fetched_page(page: FetchedPage, source: ExternalSource) -> TrustedSourceHit:
    raw = (page.body_text or "").strip()
    if not raw and page.fetch_error:
        raw = normalize_whitespace(f"Fetch note: {page.fetch_error}")
    if not raw:
        raw = "(no extractable text)"
    raw = clean_excerpt_for_evidence(raw[:8000])
    if not raw:
        raw = "(no extractable text)"
    title = (page.page_title or "").strip() or None
    return TrustedSourceHit(
        url=page.url,
        title=title,
        excerpt=raw[:8000],
        source_key=source.source_key,
        retrieved_at=page.retrieved_at,
    )


@runtime_checkable
class EvidenceExtractor(Protocol):
    """Extract bounded, attributable snippets from allowlisted page content (legacy path)."""

    def extract(
        self,
        *,
        page: FetchedPage,
        source: ExternalSource,
        query_hints: list[str],
        max_snippets: int,
    ) -> list[ExtractedSnippet]:
        """
        Produce snippets for one page.

        query_hints guides relevance ranking; must not exceed max_snippets total returned
        for this page from this call.
        """


class PageEvidenceExtractor:
    """
    Maps a single :class:`FetchedPage` into :class:`ExtractedSnippet` rows via :class:`RetrievalEvidenceExtractor`.

    ``query_hints`` are accepted for API compatibility; deterministic splitting does not rank by them yet.
    """

    def extract(
        self,
        *,
        page: FetchedPage,
        source: ExternalSource,
        query_hints: list[str],
        max_snippets: int,
    ) -> list[ExtractedSnippet]:
        _ = query_hints
        hit = _trusted_hit_from_fetched_page(page, source)
        bundle = RetrievalEvidenceExtractor.from_trusted_hits(
            [hit],
            provider_id="page_fetch",
            sources_by_key={source.source_key: source},
            max_snippets_out=16,
        )
        return bundle.research_evidence.snippets[:max_snippets]


class NullEvidenceExtractor:
    """Placeholder: no snippets for legacy ``extract``; empty structured bundles for retrieval."""

    def extract(
        self,
        *,
        page: FetchedPage,
        source: ExternalSource,
        query_hints: list[str],
        max_snippets: int,
    ) -> list[ExtractedSnippet]:
        _ = (page, source, query_hints, max_snippets)
        return []

    @staticmethod
    def empty_extraction(*, provider_id: str = "none") -> EvidenceExtractionResult:
        now = datetime.now(timezone.utc)
        fe = FrontendEvidenceBundle(citations=[], generated_at=now)
        rd = ReviewDraftEvidenceBundle(
            snippets=[],
            claims=[],
            sources=[],
            provider_id=provider_id,
            generated_at=now,
        )
        return EvidenceExtractionResult(
            frontend=fe,
            review_draft=rd,
            research_evidence=ResearchEvidence(),
        )
