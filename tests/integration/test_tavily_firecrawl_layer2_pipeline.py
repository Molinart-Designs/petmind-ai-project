"""
Integration-style tests for the Layer2 pipeline: expand → Tavily-like search → rank/dedupe → Firecrawl → extract.

Uses mocked retrieval and Firecrawl (no real HTTP). Internal RAG is out of scope here (orchestrator).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import HttpUrl

from src.research.source_registry import MedicalSensitivityLevel, TrustedSourceEntry, TrustedSourceRegistry
from src.research.trusted_research_service import (
    TrustedExternalRetrievalPort,
    TrustedResearchRequest,
    TrustedResearchService,
)
from src.research.web_retriever import (
    FetchedPage,
    TrustedRetrievalInput,
    TrustedRetrievalResult,
    TrustedSourceHit,
)


def _registry() -> TrustedSourceRegistry:
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


class _FixedQueries:
    def expand(self, question: str, **kwargs: object) -> list[str]:
        _ = (question, kwargs)
        return ["expanded nutrition query"]


class _TavilyLikeRetrieval(TrustedExternalRetrievalPort):
    """Duplicate URL with lower score first; pipeline must keep higher score and rank /other second."""

    async def retrieve(self, inp: TrustedRetrievalInput) -> TrustedRetrievalResult:  # noqa: ARG002
        now = datetime.now(timezone.utc)
        h_low = TrustedSourceHit(
            url=HttpUrl("https://example.com/page"),
            title="low",
            excerpt="low excerpt text long enough for validation rules here.",
            source_key="doc_example",
            retrieved_at=now,
            relevance_score=0.31,
        )
        h_high = TrustedSourceHit(
            url=HttpUrl("https://example.com/page"),
            title="high",
            excerpt="higher score duplicate url excerpt text long enough.",
            source_key="doc_example",
            retrieved_at=now,
            relevance_score=0.96,
        )
        h_b = TrustedSourceHit(
            url=HttpUrl("https://example.com/other"),
            title="b",
            excerpt="other path excerpt text long enough for tests.",
            source_key="doc_example",
            retrieved_at=now,
            relevance_score=0.55,
        )
        return TrustedRetrievalResult(hits=[h_low, h_high, h_b], blocked=[], provider_id="tavily")


@pytest.mark.asyncio
async def test_layer2_rank_dedupe_then_firecrawl_fetches_top_urls_in_order() -> None:
    fetched_order: list[str] = []

    class _RecordingFirecrawl:
        max_urls_per_request = 2

        async def fetch_trusted_pages(self, pages, *, registry):  # noqa: ARG002
            fetched_order.extend(str(u) for u, _ in pages)
            now = datetime.now(timezone.utc)
            return [
                FetchedPage(
                    url=u,
                    source_key=sk,
                    http_status=200,
                    content_type="text/markdown",
                    body_text="# FC\n\n" + ("Markdown body for trusted extraction pipeline. " * 5),
                    retrieved_at=now,
                    fetch_error=None,
                )
                for u, sk in pages
            ]

    reg = _registry()
    svc = TrustedResearchService(
        registry=reg,
        query_expander=_FixedQueries(),
        retrieval=_TavilyLikeRetrieval(),
        content_fetcher=_RecordingFirecrawl(),
    )

    outcome = await svc.run(
        TrustedResearchRequest(
            question="What should my dog eat?",
            max_pages_per_source=2,
        )
    )

    assert outcome.research.expanded_queries == ["expanded nutrition query"]
    assert fetched_order[0].rstrip("/").endswith("/page")
    assert fetched_order[1].rstrip("/").endswith("/other")
    assert len(fetched_order) == 2
    texts = " ".join(sn.text for sn in outcome.research.evidence.snippets)
    assert "Markdown body for trusted extraction" in texts


@pytest.mark.asyncio
async def test_max_pages_per_source_zero_skips_firecrawl() -> None:
    called = False

    class _NoFirecrawl:
        max_urls_per_request = 3

        async def fetch_trusted_pages(self, pages, *, registry):  # noqa: ARG002
            nonlocal called
            called = True
            return []

    reg = _registry()
    hit = TrustedSourceHit(
        url=HttpUrl("https://example.com/p"),
        title="t",
        excerpt="stub excerpt text long enough for tests here.",
        source_key="doc_example",
        retrieved_at=datetime.now(timezone.utc),
    )

    class _OneHitRetrieval(TrustedExternalRetrievalPort):
        async def retrieve(self, inp: TrustedRetrievalInput) -> TrustedRetrievalResult:  # noqa: ARG002
            return TrustedRetrievalResult(hits=[hit], blocked=[], provider_id="stub")

    svc = TrustedResearchService(
        registry=reg,
        query_expander=_FixedQueries(),
        retrieval=_OneHitRetrieval(),
        content_fetcher=_NoFirecrawl(),
    )
    outcome = await svc.run(
        TrustedResearchRequest(question="q?", max_pages_per_source=0),
    )
    assert called is False
    assert "stub excerpt" in outcome.research.evidence.snippets[0].text.lower()
