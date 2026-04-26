"""Firecrawl content provider: HTTP mocked; allowlist enforced."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import HttpUrl

from src.research.firecrawl_content import (
    FirecrawlContentProvider,
    build_trusted_content_fetcher,
    _dedupe_and_cap_pages,
)
from src.research.source_registry import MedicalSensitivityLevel, TrustedSourceEntry, TrustedSourceRegistry


def _registry() -> TrustedSourceRegistry:
    return TrustedSourceRegistry(
        entries=(
            TrustedSourceEntry(
                source_key="doc_example",
                allowlisted_domains=("example.com",),
                category="documentation",
                authority_score=0.7,
                medical_sensitivity=MedicalSensitivityLevel.none,
                auto_ingest_allowed=True,
            ),
        )
    )


def test_dedupe_and_cap_pages() -> None:
    u1 = HttpUrl("https://example.com/a")
    u2 = HttpUrl("https://example.com/b")
    out = _dedupe_and_cap_pages(
        [(u1, "doc_example"), (u1, "doc_example"), (u2, "doc_example")],
        max_urls=2,
    )
    assert len(out) == 2
    assert str(out[0][0]) == str(u1)


@pytest.mark.asyncio
async def test_firecrawl_scrape_success_returns_markdown_body() -> None:
    reg = _registry()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url).rstrip("/").endswith("/v2/scrape")
        body = json.loads(request.content.decode())
        assert body["url"].startswith("https://example.com")
        assert body["formats"] == ["markdown"]
        assert body["onlyMainContent"] is True
        assert body["timeout"] >= 1000
        auth = request.headers.get("authorization", "")
        assert auth.startswith("Bearer ")
        assert auth.endswith("fc-test-key")
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "markdown": "## Section\n\nClean markdown for evidence extraction. " * 3,
                },
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        prov = FirecrawlContentProvider(
            api_key="fc-test-key",
            api_base_url="https://api.firecrawl.dev",
            max_urls_per_request=2,
            timeout_s=25,
            http_client=client,
        )
        pages = await prov.fetch_trusted_pages(
            [
                (HttpUrl("https://example.com/doc"), "doc_example"),
            ],
            registry=reg,
        )

    assert len(pages) == 1
    p = pages[0]
    assert p.http_status == 200
    assert p.content_type == "text/markdown"
    assert p.body_text is not None
    assert "Clean markdown" in p.body_text
    assert p.fetch_error is None


@pytest.mark.asyncio
async def test_firecrawl_http_error_sets_fetch_error() -> None:
    reg = _registry()

    async def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(402, json={"error": "Payment required"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        prov = FirecrawlContentProvider(
            api_key="fc-bad",
            http_client=client,
            max_urls_per_request=1,
        )
        pages = await prov.fetch_trusted_pages(
            [(HttpUrl("https://example.com/x"), "doc_example")],
            registry=reg,
        )

    assert len(pages) == 1
    assert pages[0].fetch_error is not None
    assert "Payment" in pages[0].fetch_error


@pytest.mark.asyncio
async def test_firecrawl_skips_scrape_when_url_matches_denylist(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.config import settings

    monkeypatch.setattr(settings, "trusted_external_denylist", "example.com/deny-this")
    reg = _registry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
        prov = FirecrawlContentProvider(api_key="fc-key", http_client=client, max_urls_per_request=2)
        pages = await prov.fetch_trusted_pages(
            [(HttpUrl("https://example.com/deny-this/page"), "doc_example")],
            registry=reg,
        )
    assert len(pages) == 1
    assert pages[0].fetch_error == "url_denied_by_policy"
    assert pages[0].body_text is None


@pytest.mark.asyncio
async def test_firecrawl_skips_scrape_when_domain_not_allowlisted() -> None:
    reg = _registry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
        prov = FirecrawlContentProvider(api_key="fc-key", http_client=client, max_urls_per_request=2)
        pages = await prov.fetch_trusted_pages(
            [(HttpUrl("https://evil.test/nope"), "doc_example")],
            registry=reg,
        )
    assert len(pages) == 1
    assert pages[0].fetch_error == "domain_not_allowlisted"
    assert pages[0].body_text is None


@pytest.mark.asyncio
async def test_firecrawl_success_false_sets_fetch_error() -> None:
    reg = _registry()

    async def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(200, json={"success": False, "error": "blocked_by_policy"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        prov = FirecrawlContentProvider(api_key="fc", http_client=client)
        pages = await prov.fetch_trusted_pages(
            [(HttpUrl("https://example.com/z"), "doc_example")],
            registry=reg,
        )
    assert pages[0].fetch_error == "blocked_by_policy"


def test_build_trusted_content_fetcher_none_when_master_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.config import settings

    monkeypatch.setattr(settings, "enable_trusted_external_retrieval", False)
    monkeypatch.setattr(settings, "trusted_content_provider", "firecrawl")
    monkeypatch.setattr(settings, "firecrawl_api_key", "secret")
    assert build_trusted_content_fetcher() is None


def test_build_trusted_content_fetcher_none_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.config import settings

    monkeypatch.setattr(settings, "enable_trusted_external_retrieval", True)
    monkeypatch.setattr(settings, "trusted_content_provider", "firecrawl")
    monkeypatch.setattr(settings, "firecrawl_api_key", "")
    assert build_trusted_content_fetcher() is None


def test_build_trusted_content_fetcher_returns_provider_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core.config import settings

    monkeypatch.setattr(settings, "enable_trusted_external_retrieval", True)
    monkeypatch.setattr(settings, "trusted_content_provider", "firecrawl")
    monkeypatch.setattr(settings, "firecrawl_api_key", "fc-secret")
    monkeypatch.setattr(settings, "firecrawl_api_base_url", "https://api.firecrawl.dev")
    monkeypatch.setattr(settings, "firecrawl_max_urls_per_request", 2)
    monkeypatch.setattr(settings, "firecrawl_timeout_seconds", 20)
    prov = build_trusted_content_fetcher()
    assert prov is not None
    assert prov.max_urls_per_request == 2
