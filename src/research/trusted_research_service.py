"""
Orchestrates Layer 2 trusted external research using registry, expansion, fetch, and extraction.

**Pipeline (cuando el orquestador invoca L2 con flags activos):**

1. **Consultas expandidas** — ``QueryExpander`` + ``context_queries`` hacia búsqueda (p. ej. Tavily).
2. **Búsqueda allowlist** — :class:`TrustedExternalRetrievalService` (Tavily/stub): solo dominios registrados.
3. **Dedup + ranking** — :func:`dedupe_and_rank_trusted_hits` por ``relevance_score`` (Tavily) u orden estable.
4. **Fetch de contenido** — opcional :class:`~src.research.firecrawl_content.FirecrawlContentProvider` sobre las
   primeras URLs del ranking (tope por ``firecrawl_max_urls_per_request`` y ``max_pages_per_source``).
5. **Evidencia** — :class:`RetrievalEvidenceExtractor` → snippets atribuibles.
6. **``ResearchResult``** — envelope para el orquestador / persistencia L3.

**RAG interno** corre antes en :class:`~src.core.orchestrator.RAGOrchestrator`; este módulo **no** llama al
retriever interno. L2 solo se usa si ``ENABLE_TRUSTED_EXTERNAL_RETRIEVAL`` y ``ALLOW_PROVISIONAL_IN_QUERY``
(orquestador + ``DisabledTrustedResearchService`` cuando corresponde).

**Allowlist:** no hay dominios fuera de registro en ``TrustedRetrievalInput``.

**API pública:** el borrador de revisión sigue en ``EvidenceExtractionResult`` para curadores; HTTP ``/query``
fuerza ``review_draft: null`` en ``src/api/routes.py``.

Does not persist candidates; see ingest_candidates and knowledge_refresh for Layer 3.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ConfigDict, HttpUrl

from src.api.schemas import PetProfile, QueryFilters
from src.research.evidence_extractor import (
    EvidenceExtractionResult,
    RetrievalEvidenceExtractor,
    _trusted_hit_from_fetched_page,
    select_snippets_for_research_result,
)
from src.research.knowledge_refresh import default_review_after
from src.research.query_expander import DeterministicResearchQueryExpander, QueryExpander
from src.research.schemas import (
    ExternalSource,
    ExternalSourceType,
    ExtractedSnippet,
    ResearchEvidence,
    ResearchResult,
)
from src.research.source_registry import (
    TrustedSourceRegistryPort,
    build_runtime_trusted_source_registry,
)
from src.research.evidence_scope_infer import merge_research_scope
from src.research.external_ranking import normalize_provider_relevance
from src.research.http_url_utils import domain_from_http_url
from src.research.domain_authority import domain_authority_score, is_anecdotal_social_domain
from src.research.web_retriever import (
    TrustedExternalRetrievalService,
    TrustedRetrievalInput,
    TrustedRetrievalResult,
    TrustedSourceHit,
    build_external_retrieval_provider,
)
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.research.firecrawl_content import FirecrawlContentProvider

logger = get_logger(__name__)

_CONTENT_FETCHER_UNSET = object()


def dedupe_and_rank_trusted_hits(hits: list[TrustedSourceHit]) -> list[TrustedSourceHit]:
    """
    Sort by provider relevance **and** domain authority tier, then dedupe by URL keeping the best row.
    """

    def _composite(h: TrustedSourceHit) -> float:
        dom = domain_from_http_url(str(h.url))
        rel = normalize_provider_relevance(h.relevance_score)
        auth = domain_authority_score(dom)
        return 0.55 * rel + 0.45 * auth

    # Primary sort: composite relevance; tie-break: prefer non-Reddit so reddit.com is never primary.
    sorted_hits = sorted(
        hits,
        key=lambda h: (
            _composite(h),
            0 if is_anecdotal_social_domain(domain_from_http_url(str(h.url))) else 1,
        ),
        reverse=True,
    )
    out: list[TrustedSourceHit] = []
    seen: set[str] = set()
    for h in sorted_hits:
        key = str(h.url)
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


class TrustedResearchRequest(BaseModel):
    """Input to one trusted research orchestration run (Layer 2)."""

    question: str = Field(..., min_length=1, max_length=2000)
    pet_profile: PetProfile | None = None
    filters: QueryFilters | None = None
    expanded_queries: list[str] = Field(
        default_factory=list,
        description="Optional pre-expanded queries; if empty, expander may fill",
    )
    candidate_urls: list[HttpUrl] = Field(
        default_factory=list,
        description="Explicit allowlisted targets; if empty, uses each configured source base_url",
    )
    max_snippets: int = Field(default=8, ge=0, le=32)
    max_pages_per_source: int = Field(default=2, ge=0, le=10)
    review_after_days: int = Field(default=30, ge=1, le=365, description="Horizon for suggested review_after")

    model_config = ConfigDict(extra="forbid")


class TrustedResearchOutcome(BaseModel):
    """Structured output of :meth:`TrustedResearchService.run` (evidence + persistence-ready extraction)."""

    research: ResearchResult
    extraction: EvidenceExtractionResult
    debug_metrics: dict[str, Any] | None = Field(
        default=None,
        description="Optional counters for structured DEBUG logs (no secrets).",
    )

    model_config = ConfigDict(extra="forbid")


@runtime_checkable
class TrustedExternalRetrievalPort(Protocol):
    """Pluggable trusted retrieval (default: :class:`TrustedExternalRetrievalService`)."""

    async def retrieve(self, inp: TrustedRetrievalInput) -> TrustedRetrievalResult:
        """Run allowlisted retrieval and return structured hits plus policy blocks."""


def _scope_from_request(request: TrustedResearchRequest) -> tuple[str | None, str | None, str | None, str | None]:
    """Derive topic/species/breed/life_stage for the result envelope (filters override where set)."""
    topic: str | None = None
    species: str | None = None
    breed: str | None = None
    life_stage: str | None = None
    if request.pet_profile:
        species = request.pet_profile.species.strip() if request.pet_profile.species else None
        breed = request.pet_profile.breed.strip() if request.pet_profile.breed else None
        life_stage = request.pet_profile.life_stage.strip() if request.pet_profile.life_stage else None
    if request.filters:
        if request.filters.category and request.filters.category.strip():
            topic = request.filters.category.strip()
        if request.filters.species and request.filters.species.strip():
            species = request.filters.species.strip()
        if request.filters.life_stage and request.filters.life_stage.strip():
            life_stage = request.filters.life_stage.strip()
    return topic, species, breed, life_stage


def _top_authority_score(snippets: list[ExtractedSnippet], sources: list[ExternalSource]) -> float:
    if snippets:
        return max(s.authority_score for s in snippets)
    if sources:
        return max(s.authority_score for s in sources)
    return 0.0


def _top_ranking_score(snippets: list[ExtractedSnippet]) -> float:
    if not snippets:
        return 0.0
    return max((s.ranking_score or 0.0) for s in snippets)


def _external_confidence_tier(
    snippet_count: int, top_authority: float, top_ranking: float
) -> Literal["high", "medium", "low"]:
    """Blend authority tier with composite retrieval rank (not a fixed 0.5 stub)."""
    blend = 0.52 * float(top_ranking) + 0.48 * float(top_authority)
    if snippet_count >= 3 and blend >= 0.58:
        return "high"
    if snippet_count >= 2 and blend >= 0.44:
        return "medium"
    return "low"


def _substantial_firecrawl_body(body: str | None, *, min_chars: int = 32) -> bool:
    return bool(body and len(body.strip()) >= min_chars)


async def _enrich_retrieval_with_firecrawl(
    *,
    fetcher: FirecrawlContentProvider,
    registry: TrustedSourceRegistryPort,
    retrieval_out: TrustedRetrievalResult,
    sources_by_key: dict[str, ExternalSource],
    max_fetch_urls: int,
) -> tuple[TrustedRetrievalResult, int, int]:
    """
    Fetch top ``max_fetch_urls`` ranked unique URLs with Firecrawl; replace excerpts when markdown is substantial.

    Returns ``(result, pages_fetched_count, pages_with_substantial_body_count)`` for structured logging.
    """
    if max_fetch_urls <= 0:
        return retrieval_out, 0, 0

    refs: list[tuple[HttpUrl, str]] = []
    for h in retrieval_out.hits:
        if len(refs) >= max_fetch_urls:
            break
        refs.append((h.url, h.source_key))
    if not refs:
        return retrieval_out, 0, 0

    fetched = await fetcher.fetch_trusted_pages(refs, registry=registry)
    pages_fetched = len(fetched)
    pages_substantial = sum(1 for p in fetched if _substantial_firecrawl_body(p.body_text))
    by_url = {str(p.url): p for p in fetched}

    new_hits: list[TrustedSourceHit] = []
    for h in retrieval_out.hits:
        page = by_url.get(str(h.url))
        if page is None:
            new_hits.append(h)
            continue
        src = sources_by_key.get(h.source_key)
        if src is None:
            new_hits.append(h)
            continue
        if not _substantial_firecrawl_body(page.body_text):
            new_hits.append(h)
            continue
        new_hits.append(_trusted_hit_from_fetched_page(page, src))

    return (
        TrustedRetrievalResult(
            hits=new_hits,
            blocked=list(retrieval_out.blocked),
            provider_id=retrieval_out.provider_id,
        ),
        pages_fetched,
        pages_substantial,
    )


def _build_evidence_summary(
    *,
    snippet_count: int,
    source_count: int,
    blocked: int,
    provider_id: str,
    expanded_count: int,
) -> str:
    return (
        f"Trusted external retrieval ({provider_id}): {snippet_count} snippet(s) extracted from "
        f"{source_count} attributed source(s). {blocked} candidate URL(s) blocked by policy. "
        f"{expanded_count} expanded research query string(s) supplied as context (allowlisted retrieval only)."
    )


class TrustedResearchService:
    """
    External fallback orchestration: expand → search (Tavily/stub) → rank/dedupe → optional Firecrawl → extract.

    Does not persist and does not call the core RAG retriever (L1 lives in the orchestrator).
    """

    def __init__(
        self,
        *,
        registry: TrustedSourceRegistryPort,
        query_expander: QueryExpander | None = None,
        retrieval: TrustedExternalRetrievalPort | None = None,
        content_fetcher: Any = _CONTENT_FETCHER_UNSET,
    ) -> None:
        self._registry = registry
        self._query_expander = query_expander or DeterministicResearchQueryExpander()
        self._retrieval: TrustedExternalRetrievalPort = retrieval or TrustedExternalRetrievalService(
            registry=registry,
            provider=build_external_retrieval_provider(registry=registry),
        )
        if content_fetcher is _CONTENT_FETCHER_UNSET:
            from src.research.firecrawl_content import build_trusted_content_fetcher

            self._content_fetcher: FirecrawlContentProvider | None = build_trusted_content_fetcher()
        else:
            self._content_fetcher = content_fetcher

    async def run(self, request: TrustedResearchRequest) -> TrustedResearchOutcome:
        """
        Layer 2 pipeline: expand queries → allowlisted search → dedupe/rank URLs → optional Firecrawl
        on top URLs → snippet extraction → :class:`ResearchResult`.

        When ``candidate_urls`` is empty, each configured trusted source ``base_url`` seeds search targets.
        """
        # 1) Expanded queries for search providers (e.g. Tavily).
        expanded = list(
            request.expanded_queries
            or self._query_expander.expand(
                request.question,
                pet_profile=request.pet_profile,
                filters=request.filters,
            )
        )

        configured = self._registry.list_sources()
        candidate_urls = list(request.candidate_urls) if request.candidate_urls else [s.base_url for s in configured]

        # 2) Allowlisted search (Tavily / stub); no unrestricted domains.
        retrieval_input = TrustedRetrievalInput(
            candidate_urls=candidate_urls,
            context_queries=expanded,
        )
        retrieval_out = await self._retrieval.retrieve(retrieval_input)
        search_hit_count_allowlisted = len(retrieval_out.hits)
        blocked_after_search = len(retrieval_out.blocked)

        # 3) Dedupe + rank candidate URLs for downstream fetch and evidence.
        ranked_hits = dedupe_and_rank_trusted_hits(list(retrieval_out.hits))
        retrieval_out = TrustedRetrievalResult(
            hits=ranked_hits,
            blocked=list(retrieval_out.blocked),
            provider_id=retrieval_out.provider_id,
        )

        sources_by_key = {s.source_key: s for s in configured}

        # 4) Fetch top ranked URLs with Firecrawl (bounded; allowlist re-checked in provider).
        pages_fetched = 0
        pages_substantial = 0
        if self._content_fetcher is not None and request.max_pages_per_source > 0:
            max_fc = min(
                self._content_fetcher.max_urls_per_request,
                request.max_pages_per_source,
            )
            retrieval_out, pages_fetched, pages_substantial = await _enrich_retrieval_with_firecrawl(
                fetcher=self._content_fetcher,
                registry=self._registry,
                retrieval_out=retrieval_out,
                sources_by_key=sources_by_key,
                max_fetch_urls=max_fc,
            )

        # 5–6) Evidence snippets + structured ResearchResult (review_draft stripped at HTTP layer for /query).
        extraction = RetrievalEvidenceExtractor.from_retrieval_result(
            retrieval_out,
            sources_by_key=sources_by_key or None,
            query=request.question,
            max_snippets_out=None,
            max_low_tier_snippets=1,
        )
        evidence = extraction.research_evidence
        if not evidence.snippets and configured:
            evidence = ResearchEvidence(snippets=[], sources=list(configured))

        now = datetime.now(timezone.utc)
        topic, species, breed, life_stage = _scope_from_request(request)
        is_med_topic = bool(topic and "medical" in topic.lower()) or any(
            w in request.question.lower() for w in ("diagnos", "medicat", "treatment", "dosis", "emergencia")
        )
        topic, species, breed, life_stage = merge_research_scope(
            topic=topic,
            species=species,
            breed=breed,
            life_stage=life_stage,
            question=request.question,
            is_medical_topic=is_med_topic,
        )

        snippets = list(evidence.snippets)
        if species or breed or life_stage:
            snippets = [
                s.model_copy(
                    update={
                        "species": s.species or species,
                        "breed": s.breed or breed,
                        "life_stage": s.life_stage or life_stage,
                    }
                )
                for s in snippets
            ]
        if request.max_snippets > 0:
            snippets = select_snippets_for_research_result(
                snippets,
                max_total=request.max_snippets,
                max_low_tier=1,
            )
        ref_ids = {s.external_source_id for s in snippets}
        sources_filtered = [s for s in evidence.sources if s.id in ref_ids]
        evidence = ResearchEvidence(snippets=snippets, sources=sources_filtered)

        extraction_final = RetrievalEvidenceExtractor.from_research_evidence(
            evidence,
            provider_id=retrieval_out.provider_id,
            blocked_url_count=len(retrieval_out.blocked),
        )
        top_auth = _top_authority_score(evidence.snippets, evidence.sources)
        top_rank = _top_ranking_score(evidence.snippets)
        conf = _external_confidence_tier(len(evidence.snippets), top_auth, top_rank)
        summary = _build_evidence_summary(
            snippet_count=len(evidence.snippets),
            source_count=len(evidence.sources),
            blocked=len(retrieval_out.blocked),
            provider_id=retrieval_out.provider_id,
            expanded_count=len(expanded),
        )
        suggested_review = default_review_after(now, days=request.review_after_days)

        layer2_debug: dict[str, Any] = {
            "search_provider_id": retrieval_out.provider_id,
            "external_search_result_count": search_hit_count_allowlisted,
            "external_allowlisted_hit_count": search_hit_count_allowlisted,
            "external_policy_blocked_url_count": blocked_after_search,
            "external_ranked_unique_url_count": len(ranked_hits),
            "firecrawl_pages_fetched": pages_fetched,
            "firecrawl_pages_substantial_body": pages_substantial,
            "extracted_snippet_count": len(evidence.snippets),
        }
        logger.debug(
            "trusted_external_fallback_layer2",
            extra={"event": "trusted_external_fallback", "phase": "layer2", **layer2_debug},
        )

        logger.info(
            "Trusted research run complete",
            extra={
                "question_preview": request.question[:120],
                "expanded_count": len(expanded),
                "snippet_count": len(evidence.snippets),
                "blocked_urls": len(retrieval_out.blocked),
                "provider_id": retrieval_out.provider_id,
                "external_confidence": conf,
            },
        )

        research = ResearchResult(
            id=None,
            evidence=evidence,
            authority_score=top_auth,
            source_type=ExternalSourceType.allowlisted_web,
            topic=topic,
            species=species,
            breed=breed,
            life_stage=life_stage,
            retrieved_at=now,
            review_after=suggested_review,
            evidence_summary=summary,
            external_confidence=conf,
            expanded_queries=expanded,
        )
        if not evidence.snippets:
            if not configured:
                fail = "trusted_registry_empty_set_TRUSTED_EXTERNAL_ALLOWLIST_DOMAINS"
            elif search_hit_count_allowlisted == 0 and blocked_after_search > 0:
                fail = "retrieval_zero_hits_with_policy_blocks_check_provider_and_allowlist"
            elif search_hit_count_allowlisted == 0:
                fail = "retrieval_zero_hits_check_provider_api_and_allowlist"
            else:
                fail = "evidence_extraction_produced_zero_snippets"
            layer2_debug["external_failure_reason"] = fail
            logger.warning(
                "Trusted external research produced no usable snippets",
                extra={"event": "trusted_external_fallback", "phase": "layer2_no_snippets", **layer2_debug},
            )

        return TrustedResearchOutcome(
            research=research,
            extraction=extraction_final,
            debug_metrics=layer2_debug,
        )


class DisabledTrustedResearchService:
    """
    Placeholder when trusted external retrieval is inactive for this process (e.g. flags off or query disallows provisional external).

    Avoids constructing registry + retrieval stacks while keeping a uniform ``run`` contract.
    """

    async def run(self, request: TrustedResearchRequest) -> TrustedResearchOutcome:
        _ = request
        now = datetime.now(timezone.utc)
        evidence = ResearchEvidence(snippets=[], sources=[])
        research = ResearchResult(
            evidence=evidence,
            authority_score=0.0,
            source_type=ExternalSourceType.allowlisted_web,
            retrieved_at=now,
            evidence_summary="Trusted external retrieval is disabled by configuration.",
            external_confidence="low",
            expanded_queries=[],
        )
        extraction_final = RetrievalEvidenceExtractor.from_research_evidence(
            evidence,
            provider_id="disabled",
            blocked_url_count=0,
        )
        return TrustedResearchOutcome(
            research=research,
            extraction=extraction_final,
            debug_metrics={
                "external_failure_reason": "trusted_external_retrieval_disabled_by_flags",
                "extracted_snippet_count": 0,
                "external_search_result_count": 0,
                "external_allowlisted_hit_count": 0,
            },
        )


def get_trusted_research_service() -> TrustedResearchService:
    """Factory with allowlist from env and provider from ``TRUSTED_SEARCH_PROVIDER`` / ``TAVILY_API_KEY``."""
    from src.core.config import settings

    registry = build_runtime_trusted_source_registry()
    if not registry.list_sources():
        logger.warning(
            "Trusted external allowlist is empty; Layer 2 will not produce hits until domains are configured",
            extra={
                "event": "trusted_external_config",
                "hint": "Set TRUSTED_EXTERNAL_ALLOWLIST_DOMAINS (comma-separated hostnames, e.g. avma.org).",
            },
        )
    provider = build_external_retrieval_provider(registry=registry)
    retrieval = TrustedExternalRetrievalService(registry=registry, provider=provider)
    logger.info(
        "trusted_research_service_wiring",
        extra={
            "event": "trusted_research_wiring",
            "service_class": TrustedResearchService.__name__,
            "external_retrieval_provider_class": type(provider).__name__,
            "trusted_external_retrieval_service_class": TrustedExternalRetrievalService.__name__,
            "registry_configured_source_count": len(registry.list_sources()),
            "trusted_search_provider_setting": settings.trusted_search_provider.strip().lower(),
            "tavily_api_key_present": bool(settings.tavily_api_key.strip()),
            "trusted_content_provider_setting": settings.trusted_content_provider.strip().lower(),
            "firecrawl_api_key_present": bool(settings.firecrawl_api_key.strip()),
        },
    )
    return TrustedResearchService(registry=registry, retrieval=retrieval)


@lru_cache
def get_disabled_trusted_research_service() -> DisabledTrustedResearchService:
    """Singleton no-op service for orchestrator wiring when external retrieval is off."""
    return DisabledTrustedResearchService()
