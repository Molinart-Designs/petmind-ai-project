"""Tests for trusted source registry (RFC 6761 documentation host only)."""

from datetime import datetime, timezone

import pytest
from pydantic import HttpUrl

from src.research.schemas import ExternalSource, ExternalSourceType
from src.research.source_registry import (
    IngestRiskLevel,
    MedicalSensitivityLevel,
    StaticTrustedSourceRegistry,
    TrustedSourceEntry,
    TrustedSourceRegistry,
    WildcardTrustedSourceRegistry,
    build_runtime_trusted_source_registry,
)


def _example_entry(**kwargs: object) -> TrustedSourceEntry:
    defaults: dict[str, object] = {
        "source_key": "doc_example",
        "allowlisted_domains": ("example.com",),
        "category": "documentation",
        "authority_score": 0.5,
        "medical_sensitivity": MedicalSensitivityLevel.none,
        "auto_ingest_allowed": True,
    }
    defaults.update(kwargs)
    return TrustedSourceEntry(**defaults)  # type: ignore[arg-type]


def test_build_runtime_trusted_source_registry_parses_domains(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.config import settings

    monkeypatch.setattr(settings, "trusted_external_allowlist_domains", "example.com, www.foo.org ")
    reg = build_runtime_trusted_source_registry()
    assert reg.is_url_allowed("https://example.com/path") is True
    assert reg.is_url_allowed("https://sub.foo.org/y") is True


def test_wildcard_registry_allows_https_hosts() -> None:
    reg = WildcardTrustedSourceRegistry()
    assert reg.allows_any_domain() is True
    assert reg.is_url_allowed("https://unknown-example.org/path") is True
    assert reg.is_url_allowed("ftp://evil.com/") is False
    meta = reg.get_source_metadata("https://foo.bar/page")
    assert meta is not None
    assert meta.source_key == "wildcard_any_domain"


def test_build_runtime_wildcard_star(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.config import settings

    monkeypatch.setattr(settings, "trusted_external_allowlist_domains", "*,ignored.org")
    reg = build_runtime_trusted_source_registry()
    assert isinstance(reg, WildcardTrustedSourceRegistry)
    assert reg.is_url_allowed("https://anywhere.test/") is True


def test_build_runtime_trusted_source_registry_empty_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.config import settings

    monkeypatch.setattr(settings, "trusted_external_allowlist_domains", "")
    reg = build_runtime_trusted_source_registry()
    assert reg.list_sources() == []


def test_empty_registry_denies_all() -> None:
    reg = TrustedSourceRegistry()
    assert reg.is_domain_allowed("https://example.com/path") is False
    assert reg.is_url_allowed("https://example.com/") is False
    assert reg.get_source_metadata("https://example.com/") is None
    assert reg.should_auto_ingest("https://example.com/", IngestRiskLevel.low) is False


def test_subdomain_matches_allowlisted_domain() -> None:
    reg = TrustedSourceRegistry(entries=(_example_entry(),))
    assert reg.is_domain_allowed("https://sub.example.com/page") is True
    assert reg.is_url_allowed("https://sub.example.com/page") is True


def test_duplicate_domain_across_entries_raises() -> None:
    a = _example_entry(source_key="a")
    b = _example_entry(source_key="b", category="other")
    with pytest.raises(ValueError, match="domain"):
        TrustedSourceRegistry(entries=(a, b))


def test_should_auto_ingest_respects_sensitivity_and_risk() -> None:
    reg = TrustedSourceRegistry(
        entries=(
            _example_entry(
                source_key="high_sens",
                medical_sensitivity=MedicalSensitivityLevel.high,
                auto_ingest_allowed=True,
            ),
        )
    )
    url = "https://example.com/x"
    assert reg.should_auto_ingest(url, IngestRiskLevel.low) is True
    assert reg.should_auto_ingest(url, IngestRiskLevel.medium) is False
    assert reg.should_auto_ingest(url, IngestRiskLevel.high) is False


def test_get_source_metadata_fields() -> None:
    reg = TrustedSourceRegistry(entries=(_example_entry(category="gov", authority_score=0.9),))
    meta = reg.get_source_metadata("https://www.example.com/")
    assert meta is not None
    assert meta.category == "gov"
    assert meta.authority_score == 0.9
    assert meta.matched_domain == "example.com"


def test_static_registry_from_external_sources() -> None:
    ext = ExternalSource(
        id="00000000-0000-4000-8000-000000000001",
        source_key="legacy",
        base_url=HttpUrl("https://example.org/"),
        authority_score=0.4,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=datetime.now(timezone.utc),
    )
    reg = StaticTrustedSourceRegistry(sources=[ext])
    assert reg.is_domain_allowed("https://example.org/doc") is True
    assert reg.get_source("legacy") is not None
