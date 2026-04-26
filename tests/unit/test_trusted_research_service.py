"""Tests for trusted external research orchestration (mocked retrieval / expansion)."""

from datetime import datetime, timezone

import pytest
from pydantic import HttpUrl

from src.api.schemas import PetProfile, QueryFilters
from src.research.source_registry import MedicalSensitivityLevel, TrustedSourceEntry, TrustedSourceRegistry
from src.research.trusted_research_service import (
    TrustedExternalRetrievalPort,
    TrustedResearchRequest,
    TrustedResearchService,
    _build_evidence_summary,
    _external_confidence_tier,
    _scope_from_request,
    _top_authority_score,
    dedupe_and_rank_trusted_hits,
)
from src.research.web_retriever import TrustedExternalRetrievalService
from src.research.web_retriever import TrustedRetrievalInput, TrustedRetrievalResult, TrustedSourceHit


def _example_registry() -> TrustedSourceRegistry:
    return TrustedSourceRegistry(
        entries=(
            TrustedSourceEntry(
                source_key="doc_example",
                allowlisted_domains=("example.com",),
                category="documentation",
                authority_score=0.82,
                medical_sensitivity=MedicalSensitivityLevel.none,
                auto_ingest_allowed=True,
            ),
        )
    )


class _FixedQueryExpander:
    def expand(self, question: str, **kwargs: object) -> list[str]:
        _ = (question, kwargs)
        return ["alpha query", "beta query", "gamma query"]


class _FakeRetrieval(TrustedExternalRetrievalPort):
    def __init__(self, result: TrustedRetrievalResult) -> None:
        self.result = result
        self.last_input: TrustedRetrievalInput | None = None

    async def retrieve(self, inp: TrustedRetrievalInput) -> TrustedRetrievalResult:
        self.last_input = inp
        return self.result


@pytest.mark.asyncio
async def test_run_records_external_failure_reason_when_registry_empty() -> None:
    """Layer-2 run with zero configured sources yields explicit debug reason (no silent empty pipeline)."""
    reg = TrustedSourceRegistry(entries=())
    retrieval = TrustedExternalRetrievalService(registry=reg, provider=None)
    svc = TrustedResearchService(registry=reg, retrieval=retrieval, content_fetcher=None)
    outcome = await svc.run(TrustedResearchRequest(question="What is a good brushing schedule for cats?"))
    assert outcome.research.evidence.snippets == []
    assert outcome.debug_metrics is not None
    assert outcome.debug_metrics.get("external_failure_reason") == (
        "trusted_registry_empty_set_TRUSTED_EXTERNAL_ALLOWLIST_DOMAINS"
    )


@pytest.mark.asyncio
async def test_run_orchestrates_expand_retrieve_extract() -> None:
    reg = _example_registry()
    hit = TrustedSourceHit(
        url=HttpUrl("https://example.com/guide"),
        title="Guide",
        excerpt="First evidentiary sentence is long enough. Second sentence adds another snippet.",
        source_key="doc_example",
        retrieved_at=datetime.now(timezone.utc),
    )
    fake = _FakeRetrieval(TrustedRetrievalResult(hits=[hit], blocked=[], provider_id="stub"))
    svc = TrustedResearchService(
        registry=reg,
        query_expander=_FixedQueryExpander(),
        retrieval=fake,
        content_fetcher=None,
    )

    req = TrustedResearchRequest(question="What should I know about daily pet routines?")
    outcome = await svc.run(req)
    out = outcome.research

    assert fake.last_input is not None
    assert list(fake.last_input.context_queries) == ["alpha query", "beta query", "gamma query"]
    assert len(out.expanded_queries) == 3
    assert len(out.evidence.snippets) >= 1
    assert out.evidence.sources
    assert "Trusted external retrieval" in out.evidence_summary
    assert out.external_confidence in ("high", "medium", "low")
    assert out.review_after is not None
    # Registry 0.82 is blended with domain-tier baseline for example.com (not a flat pass-through).
    assert out.authority_score < 0.82
    assert out.authority_score >= 0.45
    assert out.evidence.sources[0].source_key == "doc_example"
    assert len(outcome.extraction.research_evidence.snippets) == len(out.evidence.snippets)


@pytest.mark.asyncio
async def test_scope_from_pet_and_filters() -> None:
    reg = _example_registry()
    fake = _FakeRetrieval(TrustedRetrievalResult(hits=[], blocked=[], provider_id="stub"))
    svc = TrustedResearchService(registry=reg, retrieval=fake, content_fetcher=None)
    req = TrustedResearchRequest(
        question="Question text here about nutrition",
        pet_profile=PetProfile(species="dog", breed="Beagle", life_stage="adult"),
        filters=QueryFilters(category="nutrition", species="dog", life_stage="senior"),
    )
    outcome = await svc.run(req)
    out = outcome.research
    assert out.species == "dog"
    assert out.life_stage == "senior"
    assert out.topic == "nutrition"
    assert out.breed == "Beagle"


def test_dedupe_and_rank_trusted_hits_keeps_highest_score_per_url() -> None:
    now = datetime.now(timezone.utc)
    hits = [
        TrustedSourceHit(
            url=HttpUrl("https://example.com/x"),
            title="a",
            excerpt="first duplicate excerpt long enough.",
            source_key="doc_example",
            retrieved_at=now,
            relevance_score=0.2,
        ),
        TrustedSourceHit(
            url=HttpUrl("https://example.com/x"),
            title="b",
            excerpt="winner duplicate excerpt long enough.",
            source_key="doc_example",
            retrieved_at=now,
            relevance_score=0.99,
        ),
    ]
    out = dedupe_and_rank_trusted_hits(hits)
    assert len(out) == 1
    assert out[0].relevance_score == 0.99
    assert "winner" in out[0].excerpt


def test_scope_from_request_helper() -> None:
    req = TrustedResearchRequest(
        question="q",
        pet_profile=PetProfile(species="cat"),
        filters=QueryFilters(species="rabbit"),
    )
    topic, species, breed, life_stage = _scope_from_request(req)
    assert species == "rabbit"
    assert breed is None


def test_top_authority_and_confidence_helpers() -> None:
    from src.research.schemas import ExtractedSnippet, ExternalSourceType

    now = datetime.now(timezone.utc)
    s1 = ExtractedSnippet(
        id="a",
        external_source_id="00000000-0000-4000-8000-000000000001",
        text="t",
        authority_score=0.4,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )
    s2 = ExtractedSnippet(
        id="b",
        external_source_id="00000000-0000-4000-8000-000000000001",
        text="t2",
        authority_score=0.9,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )
    assert _top_authority_score([s1, s2], []) == 0.9
    assert _external_confidence_tier(4, 0.8, 0.75) == "high"
    assert _external_confidence_tier(2, 0.6, 0.5) == "medium"
    assert _external_confidence_tier(1, 0.9, 0.3) == "low"


@pytest.mark.asyncio
async def test_run_enriches_hits_when_firecrawl_fetcher_returns_markdown() -> None:
    """Integration-style: optional Firecrawl step replaces short search excerpts with page markdown."""
    reg = _example_registry()

    class _StubFirecrawl:
        max_urls_per_request = 5

        async def fetch_trusted_pages(self, pages, *, registry):  # noqa: ARG002
            from src.research.web_retriever import FetchedPage

            return [
                FetchedPage(
                    url=u,
                    source_key=sk,
                    http_status=200,
                    content_type="text/markdown",
                    body_text="# Nutrition\n\n"
                    + (
                        "Detailed markdown paragraph for canine nutrition evidence with enough distinct words. "
                        * 5
                    ),
                    retrieved_at=datetime.now(timezone.utc),
                    fetch_error=None,
                )
                for u, sk in pages
            ]

    short_hit = TrustedSourceHit(
        url=HttpUrl("https://example.com/page"),
        title="Short",
        excerpt="Short tavily stub excerpt only.",
        source_key="doc_example",
        retrieved_at=datetime.now(timezone.utc),
    )
    fake = _FakeRetrieval(TrustedRetrievalResult(hits=[short_hit], blocked=[], provider_id="stub"))
    svc = TrustedResearchService(
        registry=reg,
        query_expander=_FixedQueryExpander(),
        retrieval=fake,
        content_fetcher=_StubFirecrawl(),
    )
    outcome = await svc.run(TrustedResearchRequest(question="dog food?"))
    texts = [sn.text for sn in outcome.research.evidence.snippets]
    assert any("Detailed markdown" in t for t in texts)
    assert any("canine nutrition" in t.lower() for t in texts)


def test_evidence_summary_includes_counts() -> None:
    s = _build_evidence_summary(
        snippet_count=2,
        source_count=1,
        blocked=3,
        provider_id="stub",
        expanded_count=5,
    )
    assert "2 snippet" in s
    assert "3 candidate URL" in s
    assert "5 expanded" in s
