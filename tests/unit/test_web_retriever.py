"""Tests for trusted web retrieval (RFC 6761 example.com only)."""

from datetime import datetime, timezone

import pytest
from pydantic import HttpUrl

from src.research.source_registry import MedicalSensitivityLevel, TrustedSourceEntry, TrustedSourceRegistry
from src.research.web_retriever import (
    NullWebRetriever,
    StubExternalRetrievalProvider,
    TrustedExternalRetrievalService,
    TrustedRetrievalInput,
    TrustedRetrievalProviderRequest,
    TrustedRetrievalTarget,
    TrustedSourceHit,
    excerpt_looks_like_html,
    partition_urls_by_allowlist,
    sanitize_provider_hits,
    url_has_allowed_retrieval_scheme,
    validate_trusted_source_hit,
)


def _example_registry() -> TrustedSourceRegistry:
    entry = TrustedSourceEntry(
        source_key="doc_example",
        allowlisted_domains=("example.com",),
        category="documentation",
        authority_score=0.5,
        medical_sensitivity=MedicalSensitivityLevel.none,
        auto_ingest_allowed=True,
    )
    return TrustedSourceRegistry(entries=(entry,))


def test_partition_allowlists_and_blocks() -> None:
    reg = _example_registry()
    urls = [
        HttpUrl("https://example.com/page"),
        HttpUrl("https://evil.test/nope"),
    ]
    allowed, blocked = partition_urls_by_allowlist(urls, reg)
    assert len(allowed) == 1
    assert allowed[0].source_key == "doc_example"
    reasons = {b.reason for b in blocked}
    assert "domain_not_allowlisted" in reasons


def test_partition_denylist_takes_precedence_over_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.config import settings

    monkeypatch.setattr(settings, "trusted_external_denylist", "example.com")
    reg = _example_registry()
    allowed, blocked = partition_urls_by_allowlist([HttpUrl("https://sub.example.com/page")], reg)
    assert allowed == []
    assert len(blocked) == 1
    assert blocked[0].reason == "url_denied_by_policy"


def test_partition_path_denylist_allows_other_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.config import settings

    monkeypatch.setattr(settings, "trusted_external_denylist", "example.com/forbidden")
    reg = _example_registry()
    allowed_ok, _ = partition_urls_by_allowlist([HttpUrl("https://example.com/allowed")], reg)
    assert len(allowed_ok) == 1
    allowed_bad, blocked = partition_urls_by_allowlist([HttpUrl("https://example.com/forbidden/x")], reg)
    assert allowed_bad == []
    assert blocked[0].reason == "url_denied_by_policy"


def test_sanitize_drops_denied_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.config import settings

    monkeypatch.setattr(settings, "trusted_external_denylist", "example.com")
    reg = _example_registry()
    hits = [
        TrustedSourceHit(
            url=HttpUrl("https://example.com/x"),
            excerpt="word " * 20,
            source_key="doc_example",
            retrieved_at=datetime.now(timezone.utc),
        ),
    ]
    kept, dropped = sanitize_provider_hits(hits, registry=reg)
    assert kept == []
    assert dropped[0].reason == "post_provider_url_denied_by_policy"


def test_url_has_allowed_retrieval_scheme() -> None:
    assert url_has_allowed_retrieval_scheme("https://example.com/x") is True
    assert url_has_allowed_retrieval_scheme("ftp://example.com/resource") is False


def test_partition_deduplicates_urls() -> None:
    reg = _example_registry()
    u = HttpUrl("https://example.com/a")
    allowed, _ = partition_urls_by_allowlist([u, u], reg)
    assert len(allowed) == 1


def test_excerpt_looks_like_html() -> None:
    assert excerpt_looks_like_html("<div>oops</div>") is True
    assert excerpt_looks_like_html("Plain text about pet care.") is False


def test_validate_trusted_source_hit() -> None:
    good = TrustedSourceHit(
        url=HttpUrl("https://example.com/"),
        title="t",
        excerpt="Plain excerpt.",
        source_key="doc_example",
        retrieved_at=datetime.now(timezone.utc),
    )
    assert validate_trusted_source_hit(good) is None
    bad = good.model_copy(update={"excerpt": "<p>html</p>"})
    assert validate_trusted_source_hit(bad) == "excerpt_resembles_html"


@pytest.mark.asyncio
async def test_stub_provider_returns_plain_text_excerpts() -> None:
    reg = _example_registry()
    stub = StubExternalRetrievalProvider()
    url = HttpUrl("https://sub.example.com/doc")
    res = await stub.retrieve(
        TrustedRetrievalProviderRequest(
            targets=(
                TrustedRetrievalTarget(
                    url=url,
                    source_key="doc_example",
                ),
            )
        )
    )
    assert res.provider_id == "stub"
    assert len(res.hits) == 1
    assert "<" not in res.hits[0].excerpt
    assert res.hits[0].source_key == "doc_example"


@pytest.mark.asyncio
async def test_trusted_service_merges_blocks_and_hits() -> None:
    reg = _example_registry()
    svc = TrustedExternalRetrievalService(registry=reg)
    inp = TrustedRetrievalInput(
        candidate_urls=[
            HttpUrl("https://example.com/ok"),
            HttpUrl("https://blocked.example/not-ok"),
        ],
        context_queries=["hint for future search"],
    )
    out = await svc.retrieve(inp)
    assert len(out.hits) == 1
    assert out.hits[0].url == HttpUrl("https://example.com/ok")
    assert len(out.blocked) == 1
    assert out.blocked[0].reason == "domain_not_allowlisted"
    assert out.provider_id == "stub"


def test_sanitize_drops_non_allowlisted_host_even_if_source_key_matches() -> None:
    """Proveedor malicioso no puede colar un host distinto del target allowlistado."""
    reg = _example_registry()
    hits = [
        TrustedSourceHit(
            url=HttpUrl("https://evil.test/phishing"),
            excerpt="word " * 20,
            source_key="doc_example",
            retrieved_at=datetime.now(timezone.utc),
        ),
    ]
    kept, dropped = sanitize_provider_hits(hits, registry=reg)
    assert kept == []
    assert len(dropped) == 1
    assert dropped[0].reason == "post_provider_not_allowlisted"


def test_sanitize_drops_html_excerpts() -> None:
    reg = _example_registry()
    url = HttpUrl("https://example.com/x")
    hits = [
        TrustedSourceHit(
            url=url,
            excerpt="ok",
            source_key="doc_example",
            retrieved_at=datetime.now(timezone.utc),
        ),
        TrustedSourceHit(
            url=url,
            excerpt="<html>bad</html>",
            source_key="doc_example",
            retrieved_at=datetime.now(timezone.utc),
        ),
    ]
    kept, dropped = sanitize_provider_hits(hits, registry=reg)
    assert len(kept) == 1
    assert len(dropped) == 1
    assert "hit_validation" in dropped[0].reason


@pytest.mark.asyncio
async def test_null_web_retriever_fetch_page() -> None:
    ret = NullWebRetriever()
    page = await ret.fetch_page(url=HttpUrl("https://example.com/"), source_key="doc_example")
    assert page.body_text is None
    assert page.fetch_error is not None
