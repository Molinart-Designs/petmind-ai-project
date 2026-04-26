"""Tavily search provider: HTTP mocked, allowlist + payload shape enforced."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import HttpUrl

from src.research.source_registry import (
    MedicalSensitivityLevel,
    TrustedSourceEntry,
    TrustedSourceRegistry,
    WildcardTrustedSourceRegistry,
)
from src.research.web_retriever import (
    TavilySearchProvider,
    TrustedExternalRetrievalService,
    TrustedRetrievalInput,
    TrustedRetrievalProviderRequest,
    TrustedRetrievalTarget,
    build_external_retrieval_provider,
)


def _registry() -> TrustedSourceRegistry:
    return TrustedSourceRegistry(
        entries=(
            TrustedSourceEntry(
                source_key="doc_example",
                allowlisted_domains=("example.com",),
                category="documentation",
                authority_score=0.6,
                medical_sensitivity=MedicalSensitivityLevel.none,
                auto_ingest_allowed=True,
            ),
        )
    )


@pytest.mark.asyncio
async def test_tavily_wildcard_registry_omits_include_domains_from_request_json() -> None:
    """When allowlist is ``*``, Tavily runs without ``include_domains`` (global search); hits still filtered by registry."""
    reg = WildcardTrustedSourceRegistry()
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        captured["body"] = body
        assert "include_domains" not in body
        assert body["api_key"] == "tvly-test"
        return httpx.Response(
            200,
            json={
                "query": body["query"],
                "results": [
                    {
                        "title": "Other site",
                        "url": "https://other-example.org/pets",
                        "content": "Cats and dogs need fresh water daily for good hydration habits.",
                        "score": 0.88,
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        prov = TavilySearchProvider(
            api_key="tvly-test",
            registry=reg,
            max_results=5,
            http_client=client,
        )
        out = await prov.retrieve(
            TrustedRetrievalProviderRequest(
                targets=(
                    TrustedRetrievalTarget(
                        url=HttpUrl("https://example.com/seed"),
                        source_key="wildcard_any_domain",
                    ),
                ),
                context_queries=("pet hydration",),
            )
        )

    assert len(out.hits) == 1
    assert "other-example.org" in str(out.hits[0].url)
    assert "fresh water" in out.hits[0].excerpt.lower()


@pytest.mark.asyncio
async def test_tavily_provider_posts_include_domains_max_results_topic_and_maps_hits() -> None:
    reg = _registry()
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url).rstrip("/").endswith("/search")
        body = json.loads(request.content.decode())
        captured["body"] = body
        assert body["include_domains"] == ["example.com"]
        assert body["max_results"] == 4
        assert body["include_answer"] is False
        assert body["topic"] == "general"
        assert "dog" in str(body["query"]).lower()
        assert body["api_key"] == "tvly-test"
        return httpx.Response(
            200,
            json={
                "query": body["query"],
                "results": [
                    {
                        "title": "Feeding guide",
                        "url": "https://www.example.com/pet/nutrition",
                        "content": "Adult dogs benefit from scheduled meals and fresh water daily.",
                        "score": 0.91,
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        prov = TavilySearchProvider(
            api_key="tvly-test",
            registry=reg,
            max_results=4,
            search_depth="basic",
            topic="general",
            base_url="https://api.tavily.com",
            http_client=client,
        )
        out = await prov.retrieve(
            TrustedRetrievalProviderRequest(
                targets=(
                    TrustedRetrievalTarget(
                        url=HttpUrl("https://example.com/seed"),
                        source_key="doc_example",
                    ),
                ),
                context_queries=("nutrition for adult dogs",),
            )
        )

    assert out.provider_id == "tavily"
    assert len(out.hits) == 1
    assert out.hits[0].source_key == "doc_example"
    assert "example.com" in str(out.hits[0].url)
    assert "scheduled meals" in out.hits[0].excerpt.lower()


@pytest.mark.asyncio
async def test_tavily_provider_http_error_yields_blocked_not_hits() -> None:
    reg = _registry()

    async def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(401, json={"detail": "unauthorized"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        prov = TavilySearchProvider(
            api_key="tvly-bad",
            registry=reg,
            http_client=client,
        )
        out = await prov.retrieve(
            TrustedRetrievalProviderRequest(
                targets=(
                    TrustedRetrievalTarget(url=HttpUrl("https://example.com/"), source_key="doc_example"),
                ),
                context_queries=("q",),
            )
        )

    assert out.hits == []
    assert any("tavily_http_401" in b.reason for b in out.blocked)


@pytest.mark.asyncio
async def test_tavily_provider_skips_non_allowlisted_result_urls() -> None:
    reg = _registry()

    async def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(
            200,
            json={
                "query": "q",
                "results": [
                    {
                        "title": "Evil",
                        "url": "https://evil.test/page",
                        "content": "Should never become a hit with plain text long enough.",
                    },
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        prov = TavilySearchProvider(api_key="tvly-test", registry=reg, http_client=client)
        out = await prov.retrieve(
            TrustedRetrievalProviderRequest(
                targets=(
                    TrustedRetrievalTarget(url=HttpUrl("https://example.com/"), source_key="doc_example"),
                ),
                context_queries=("topic",),
            )
        )

    assert out.hits == []
    assert any("tavily_result_domain_not_allowlisted" in b.reason for b in out.blocked)


@pytest.mark.asyncio
async def test_tavily_provider_skips_denylisted_result_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.config import settings

    monkeypatch.setattr(settings, "trusted_external_denylist", "example.com/bad-section")
    reg = _registry()

    async def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(
            200,
            json={
                "query": "q",
                "results": [
                    {
                        "title": "Denied path",
                        "url": "https://sub.example.com/bad-section/article",
                        "content": "Plain text excerpt about pet hydration. " * 3,
                    },
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        prov = TavilySearchProvider(api_key="tvly-test", registry=reg, http_client=client)
        out = await prov.retrieve(
            TrustedRetrievalProviderRequest(
                targets=(
                    TrustedRetrievalTarget(url=HttpUrl("https://example.com/"), source_key="doc_example"),
                ),
                context_queries=("topic",),
            )
        )

    assert out.hits == []
    assert any("tavily_result_url_denied_by_policy" in b.reason for b in out.blocked)


@pytest.mark.asyncio
async def test_trusted_external_service_wraps_tavily_and_keeps_allowlist() -> None:
    reg = _registry()

    async def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(
            200,
            json={
                "query": "q",
                "results": [
                    {
                        "title": "OK",
                        "url": "https://sub.example.com/article",
                        "content": "Plain text excerpt about pet hydration. " * 3,
                    },
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        prov = TavilySearchProvider(api_key="tvly-test", registry=reg, http_client=client)
        svc = TrustedExternalRetrievalService(registry=reg, provider=prov)
        out = await svc.retrieve(
            TrustedRetrievalInput(
                candidate_urls=[HttpUrl("https://example.com/")],
                context_queries=["hydration"],
            )
        )

    assert out.provider_id == "tavily"
    assert len(out.hits) == 1
    assert out.hits[0].source_key == "doc_example"


def test_build_external_retrieval_provider_stub_when_master_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.config import settings

    monkeypatch.setattr(settings, "enable_trusted_external_retrieval", False)
    monkeypatch.setattr(settings, "trusted_search_provider", "tavily")
    monkeypatch.setattr(settings, "tavily_api_key", "ignored")
    prov = build_external_retrieval_provider(registry=_registry())
    assert prov.provider_id == "stub"


def test_build_external_retrieval_provider_stub_when_tavily_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core.config import settings

    monkeypatch.setattr(settings, "enable_trusted_external_retrieval", True)
    monkeypatch.setattr(settings, "trusted_search_provider", "tavily")
    monkeypatch.setattr(settings, "tavily_api_key", "")
    prov = build_external_retrieval_provider(registry=_registry())
    assert prov.provider_id == "stub"


def test_build_external_retrieval_provider_returns_tavily_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core.config import settings

    monkeypatch.setattr(settings, "enable_trusted_external_retrieval", True)
    monkeypatch.setattr(settings, "trusted_search_provider", "tavily")
    monkeypatch.setattr(settings, "tavily_api_key", "tvly-secret")
    monkeypatch.setattr(settings, "tavily_max_results", 3)
    monkeypatch.setattr(settings, "tavily_search_depth", "basic")
    monkeypatch.setattr(settings, "tavily_topic", "general")
    monkeypatch.setattr(settings, "tavily_api_base_url", "https://api.tavily.com")
    prov = build_external_retrieval_provider(registry=_registry())
    assert prov.provider_id == "tavily"
