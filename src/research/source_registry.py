"""
Registry of allowlisted trusted sources.

Layer 2 must only fetch URLs whose hosts match configured domains. Configuration is supplied
explicitly via ``TrustedSourceEntry`` values (empty by default); extend by passing entries or
subclassing ``TrustedSourceRegistry``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, Field, ConfigDict, HttpUrl, field_validator

from src.research.schemas import ExternalSource, ExternalSourceType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MedicalSensitivityLevel(StrEnum):
    """How medically sensitive content from this origin may be."""

    none = "none"
    low = "low"
    moderate = "moderate"
    high = "high"


class IngestRiskLevel(StrEnum):
    """Caller-reported risk tier (e.g. from guardrails) for auto-ingestion decisions."""

    low = "low"
    medium = "medium"
    high = "high"


class TrustedSourceEntry(BaseModel):
    """
    One configurable trusted origin: domains, taxonomy, trust, sensitivity, ingestion policy.

    Domains are host labels only (no scheme or path). ``www.`` is stripped on normalize.
    """

    source_key: str = Field(..., min_length=1, max_length=64)
    allowlisted_domains: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description="Hostnames allowed for this source (no URLs, no paths)",
    )
    category: str = Field(..., min_length=1, max_length=100, description="Taxonomy label, e.g. government, vendor")
    authority_score: float = Field(..., ge=0.0, le=1.0)
    medical_sensitivity: MedicalSensitivityLevel
    auto_ingest_allowed: bool = Field(
        ...,
        description="Whether this origin may feed automatic candidate ingestion when policy allows",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("allowlisted_domains", mode="before")
    @classmethod
    def _normalize_domains(cls, value: object) -> tuple[str, ...]:
        if value is None:
            raise ValueError("allowlisted_domains is required")
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, (list, tuple)):
            items = list(value)
        else:
            raise TypeError("allowlisted_domains must be a sequence of host strings")
        out: list[str] = []
        for raw in items:
            d = str(raw).strip().lower().rstrip(".")
            if not d:
                raise ValueError("empty domain in allowlisted_domains")
            if "/" in d or "://" in d or " " in d:
                raise ValueError(f"invalid domain (use hostname only): {raw!r}")
            if d.startswith("www."):
                d = d[4:]
            out.append(d)
        return tuple(out)


class TrustedSourceMetadata(BaseModel):
    """Resolved metadata for a URL that matched a registry entry (no network I/O)."""

    source_key: str
    category: str
    authority_score: float = Field(..., ge=0.0, le=1.0)
    medical_sensitivity: MedicalSensitivityLevel
    auto_ingest_allowed: bool
    matched_domain: str = Field(..., description="Which allowlisted hostname matched the URL host")

    model_config = ConfigDict(extra="forbid", frozen=True)


def _parse_url_host(url: HttpUrl | str) -> str | None:
    """Extract normalized hostname from a full URL or bare host string."""
    s = str(url).strip()
    if not s:
        return None
    if "://" not in s:
        s = f"https://{s}"
    parsed = urlparse(s)
    host = (parsed.hostname or "").lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return host or None


def _host_matches_allowlisted_domain(host: str, domain: str) -> bool:
    """True if host equals domain or is a subdomain of domain."""
    return host == domain or host.endswith(f".{domain}")


def _resolve_entry_for_host(
    host: str,
    ordered_entries: tuple[TrustedSourceEntry, ...],
) -> tuple[TrustedSourceEntry, str] | None:
    """Return (entry, matched_domain) for the first entry whose domain list matches host."""
    for entry in ordered_entries:
        for domain in entry.allowlisted_domains:
            if _host_matches_allowlisted_domain(host, domain):
                return entry, domain
    return None


def _stable_source_id(source_key: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"petmind:trusted-source:{source_key}"))


def _entry_to_external_source(entry: TrustedSourceEntry, *, retrieved_at: datetime | None = None) -> ExternalSource:
    """Materialize a runtime ``ExternalSource`` row for RAG/evidence wiring."""
    when = retrieved_at or datetime.now(timezone.utc)
    primary = sorted(entry.allowlisted_domains)[0]
    base = f"https://{primary}/"
    return ExternalSource(
        id=_stable_source_id(entry.source_key),
        source_key=entry.source_key,
        base_url=HttpUrl(base),
        authority_score=entry.authority_score,
        source_type=ExternalSourceType.allowlisted_web,
        topic=None,
        species=None,
        breed=None,
        life_stage=None,
        retrieved_at=when,
        review_after=None,
    )


def _entries_from_external_sources(sources: list[ExternalSource]) -> tuple[TrustedSourceEntry, ...]:
    """Best-effort mapping for legacy ``ExternalSource`` lists (conservative ingestion defaults)."""
    out: list[TrustedSourceEntry] = []
    for src in sources:
        host = _parse_url_host(str(src.base_url))
        if not host:
            continue
        out.append(
            TrustedSourceEntry(
                source_key=src.source_key,
                allowlisted_domains=(host,),
                category="legacy",
                authority_score=src.authority_score,
                medical_sensitivity=MedicalSensitivityLevel.moderate,
                auto_ingest_allowed=False,
            )
        )
    return tuple(out)


@runtime_checkable
class TrustedSourceRegistryPort(Protocol):
    """Dependency surface for trusted-source resolution (tests may provide fakes)."""

    def list_sources(self) -> list[ExternalSource]:
        """Return one ``ExternalSource`` snapshot per configured logical source."""

    def get_source(self, source_key: str) -> ExternalSource | None:
        """Return materialized source by key."""

    def is_url_allowed(self, url: HttpUrl | str) -> bool:
        """True if URL scheme is http(s) and host matches an allowlisted domain."""

    def is_domain_allowed(self, url: HttpUrl | str) -> bool:
        """True if the URL's host matches any allowlisted domain (ignores path)."""

    def get_source_metadata(self, url: HttpUrl | str) -> TrustedSourceMetadata | None:
        """Return metadata for the first registry entry matching the URL host, or None."""

    def should_auto_ingest(self, url: HttpUrl | str, risk_level: IngestRiskLevel | str) -> bool:
        """Whether auto-ingestion is allowed for this URL given caller risk and source policy."""


class TrustedSourceRegistry:
    """
    In-memory trusted source registry built from explicit ``TrustedSourceEntry`` rows.

    Starts empty unless entries are provided. Duplicate domains across entries are rejected
    at construction to keep resolution deterministic.
    """

    def __init__(self, entries: tuple[TrustedSourceEntry, ...] | list[TrustedSourceEntry] | None = None) -> None:
        self._entries: tuple[TrustedSourceEntry, ...] = tuple(entries or ())
        self._by_key: dict[str, TrustedSourceEntry] = {}
        seen_domains: dict[str, str] = {}
        for entry in self._entries:
            if entry.source_key in self._by_key:
                raise ValueError(f"duplicate source_key in trusted registry: {entry.source_key!r}")
            self._by_key[entry.source_key] = entry
            for domain in entry.allowlisted_domains:
                if domain in seen_domains:
                    raise ValueError(
                        f"domain {domain!r} is registered for both {seen_domains[domain]!r} and {entry.source_key!r}"
                    )
                seen_domains[domain] = entry.source_key
        logger.debug(
            "TrustedSourceRegistry initialized",
            extra={"entry_count": len(self._entries), "domain_count": len(seen_domains)},
        )

    def list_sources(self) -> list[ExternalSource]:
        """Return materialized external sources for all entries (stable ids per source_key)."""
        return [_entry_to_external_source(e) for e in self._entries]

    def get_source(self, source_key: str) -> ExternalSource | None:
        """Return ``ExternalSource`` for a key, or None."""
        entry = self._by_key.get(source_key)
        if entry is None:
            return None
        return _entry_to_external_source(entry)

    def is_domain_allowed(self, url: HttpUrl | str) -> bool:
        """Return True when the URL host matches at least one allowlisted domain."""
        host = _parse_url_host(url)
        if not host:
            return False
        return _resolve_entry_for_host(host, self._entries) is not None

    def is_url_allowed(self, url: HttpUrl | str) -> bool:
        """Require http(s) and an allowlisted host (path is not restricted beyond scheme/host)."""
        s = str(url).strip()
        if not s:
            return False
        if "://" not in s:
            s = f"https://{s}"
        parsed = urlparse(s)
        if parsed.scheme not in ("http", "https"):
            return False
        return self.is_domain_allowed(url)

    def get_source_metadata(self, url: HttpUrl | str) -> TrustedSourceMetadata | None:
        """Return structured metadata for the matching entry, or None if host is not allowlisted."""
        host = _parse_url_host(url)
        if not host:
            return None
        resolved = _resolve_entry_for_host(host, self._entries)
        if resolved is None:
            return None
        entry, matched_domain = resolved
        return TrustedSourceMetadata(
            source_key=entry.source_key,
            category=entry.category,
            authority_score=entry.authority_score,
            medical_sensitivity=entry.medical_sensitivity,
            auto_ingest_allowed=entry.auto_ingest_allowed,
            matched_domain=matched_domain,
        )

    def should_auto_ingest(self, url: HttpUrl | str, risk_level: IngestRiskLevel | str) -> bool:
        """
        Combine ``auto_ingest_allowed`` with medical sensitivity and caller ``risk_level``.

        Conservative: high medical sensitivity never auto-ingests when risk is high; high
        sensitivity only allows auto-ingest at low caller risk.
        """
        meta = self.get_source_metadata(url)
        if meta is None or not meta.auto_ingest_allowed:
            return False
        risk = IngestRiskLevel(risk_level) if isinstance(risk_level, str) else risk_level
        sens = meta.medical_sensitivity
        if sens in (MedicalSensitivityLevel.none, MedicalSensitivityLevel.low):
            return True
        if sens == MedicalSensitivityLevel.moderate:
            return risk != IngestRiskLevel.high
        return risk == IngestRiskLevel.low

    def allows_any_domain(self) -> bool:
        return False


_WILDCARD_SOURCE_KEY = "wildcard_any_domain"
_WILDCARD_SEED_HOST = "example.com"


class WildcardTrustedSourceRegistry:
    """
    Allowlist ``*``: acepta cualquier host ``http``/``https`` con nombre no vacío.

    Pensado solo para entornos controlados o depuración: anula la política de dominios fijos.
    Sigue sin permitir esquemas distintos de http(s). ``auto_ingest_allowed`` es siempre false.
    """

    def allows_any_domain(self) -> bool:
        return True

    def list_sources(self) -> list[ExternalSource]:
        when = datetime.now(timezone.utc)
        return [
            ExternalSource(
                id=_stable_source_id(_WILDCARD_SOURCE_KEY),
                source_key=_WILDCARD_SOURCE_KEY,
                base_url=HttpUrl(f"https://{_WILDCARD_SEED_HOST}/"),
                authority_score=0.5,
                source_type=ExternalSourceType.allowlisted_web,
                retrieved_at=when,
            )
        ]

    def get_source(self, source_key: str) -> ExternalSource | None:
        for src in self.list_sources():
            if src.source_key == source_key:
                return src
        return None

    def is_domain_allowed(self, url: HttpUrl | str) -> bool:
        return _parse_url_host(url) is not None

    def is_url_allowed(self, url: HttpUrl | str) -> bool:
        s = str(url).strip()
        if not s:
            return False
        if "://" not in s:
            s = f"https://{s}"
        parsed = urlparse(s)
        if parsed.scheme not in ("http", "https"):
            return False
        return bool(parsed.hostname)

    def get_source_metadata(self, url: HttpUrl | str) -> TrustedSourceMetadata | None:
        if not self.is_url_allowed(url):
            return None
        host = _parse_url_host(url) or ""
        return TrustedSourceMetadata(
            source_key=_WILDCARD_SOURCE_KEY,
            category="wildcard_allow_all",
            authority_score=0.55,
            medical_sensitivity=MedicalSensitivityLevel.high,
            auto_ingest_allowed=False,
            matched_domain=host or "*",
        )

    def should_auto_ingest(self, url: HttpUrl | str, risk_level: IngestRiskLevel | str) -> bool:
        return False


class StaticTrustedSourceRegistry(TrustedSourceRegistry):
    """
    Backwards-compatible registry that accepts legacy ``ExternalSource`` runtime rows.

    New code should prefer ``TrustedSourceRegistry(entries=...)`` with explicit
    ``TrustedSourceEntry`` definitions.
    """

    def __init__(self, sources: list[ExternalSource] | None = None) -> None:
        super().__init__(entries=_entries_from_external_sources(sources or []))


def build_runtime_trusted_source_registry() -> TrustedSourceRegistry | WildcardTrustedSourceRegistry:
    """
    Build the process allowlist for Layer 2 from ``TRUSTED_EXTERNAL_ALLOWLIST_DOMAINS``.

    Hostnames only (comma-separated). Use a single ``*`` (optionally among commas) to allow **any**
    ``http``/``https`` hostname (modo de alto riesgo; solo entornos controlados).

    Empty string → empty registry (no candidate URLs until configured).
    """
    from src.core.config import settings

    raw = (getattr(settings, "trusted_external_allowlist_domains", None) or "").strip()
    if not raw:
        return TrustedSourceRegistry(entries=())

    domains: list[str] = []
    has_wildcard = False
    for part in raw.split(","):
        p = part.strip().lower().rstrip(".")
        if not p:
            continue
        if p == "*":
            has_wildcard = True
            continue
        if "/" in p or "://" in p or " " in p:
            logger.warning(
                "Skipping invalid trusted allowlist domain (use hostname only)",
                extra={"segment_preview": part[:120]},
            )
            continue
        if p.startswith("www."):
            p = p[4:]
        if p not in domains:
            domains.append(p)

    if has_wildcard:
        if domains:
            logger.warning(
                "TRUSTED_EXTERNAL_ALLOWLIST_DOMAINS contains '*'; ignoring other host entries and allowing any http(s) host",
                extra={"ignored_domain_count": len(domains)},
            )
        logger.warning(
            "Trusted external allowlist is wildcard (*): any http(s) domain is permitted for Layer 2",
            extra={"event": "trusted_external_wildcard_allowlist"},
        )
        return WildcardTrustedSourceRegistry()

    if not domains:
        return TrustedSourceRegistry(entries=())

    entry = TrustedSourceEntry(
        source_key="env_configured_allowlist",
        allowlisted_domains=tuple(domains),
        category="env_allowlisted",
        authority_score=0.65,
        medical_sensitivity=MedicalSensitivityLevel.moderate,
        auto_ingest_allowed=False,
    )
    return TrustedSourceRegistry(entries=(entry,))
