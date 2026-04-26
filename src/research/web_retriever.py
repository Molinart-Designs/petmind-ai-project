"""
Trusted external retrieval: allowlisted targets only, structured hits, pluggable providers.

**Cadena de política URL (denylist primero, luego allowlist):**

1. ``partition_urls_by_allowlist`` — ``TRUSTED_EXTERNAL_DENYLIST`` bloquea antes que la allowlist;
   solo ``TrustedRetrievalTarget`` cuyas URLs pasan ``registry.is_url_allowed`` y resuelven metadatos;
   el resto va a ``BlockedUrl``.
2. El proveedor (:class:`ExternalRetrievalProvider`) recibe **únicamente** esos targets.
3. ``sanitize_provider_hits`` — defensa en profundidad: descarta hits denegados, no allowlist,
   o que fallen ``validate_trusted_source_hit``.

Los proveedores no deben “descubrir” URLs nuevas por su cuenta; cualquier backend real debe
limitarse a los ``targets`` validados (sin crawling abierto).

Low-level :class:`WebRetriever` / :class:`FetchedPage` siguen disponibles para pipelines de una URL.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Literal, Protocol, cast, runtime_checkable
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field, ConfigDict, HttpUrl, TypeAdapter

from src.research.source_registry import TrustedSourceRegistryPort
from src.research.url_denylist import get_trusted_url_denylist
from src.utils.logger import get_logger

logger = get_logger(__name__)


def url_has_allowed_retrieval_scheme(url: HttpUrl | str) -> bool:
    """True only for ``http`` and ``https`` (used before allowlist checks)."""
    return urlparse(str(url).strip()).scheme.lower() in ("http", "https")


# ---------------------------------------------------------------------------
# Structured retrieval (provider-agnostic)
# ---------------------------------------------------------------------------


class BlockedUrl(BaseModel):
    """A candidate URL that was not retrieved because of policy or validation."""

    url: str = Field(..., max_length=2000)
    reason: str = Field(..., max_length=500)

    model_config = ConfigDict(extra="forbid", frozen=True)


class TrustedSourceHit(BaseModel):
    """
    One normalized retrieval hit: no raw HTML payload.

    ``excerpt`` must be plain text suitable for evidence extraction downstream.
    ``relevance_score`` is optional (e.g. Tavily); used to rank before dedupe and Firecrawl fetch.
    """

    url: HttpUrl
    title: str | None = Field(default=None, max_length=500)
    excerpt: str = Field(..., min_length=1, max_length=8000, description="Plain text snippet only")
    source_key: str = Field(..., min_length=1, max_length=64)
    retrieved_at: datetime
    relevance_score: float | None = Field(
        default=None,
        description="Provider relevance (e.g. Tavily); higher ranks first for dedupe/Firecrawl cap.",
    )

    model_config = ConfigDict(extra="forbid")


class TrustedRetrievalResult(BaseModel):
    """Outcome of a trusted retrieval batch (hits plus policy blocks)."""

    hits: list[TrustedSourceHit] = Field(default_factory=list)
    blocked: list[BlockedUrl] = Field(default_factory=list)
    provider_id: str = Field(..., min_length=1, max_length=64)

    model_config = ConfigDict(extra="forbid")


class TrustedRetrievalInput(BaseModel):
    """
    Caller input: explicit URL targets plus optional query strings for future search backends.

    ``context_queries`` are not used by the stub provider; real search providers may consume them.
    """

    candidate_urls: list[HttpUrl] = Field(default_factory=list)
    context_queries: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class TrustedRetrievalTarget(BaseModel):
    """Single allowlisted URL with resolved ``source_key`` (passed to providers)."""

    url: HttpUrl
    source_key: str = Field(..., min_length=1, max_length=64)

    model_config = ConfigDict(extra="forbid", frozen=True)


class TrustedRetrievalProviderRequest(BaseModel):
    """Validated payload sent to an :class:`ExternalRetrievalProvider`."""

    targets: tuple[TrustedRetrievalTarget, ...] = Field(default_factory=tuple)
    context_queries: tuple[str, ...] = Field(default_factory=tuple)

    model_config = ConfigDict(extra="forbid")


def excerpt_looks_like_html(text: str) -> bool:
    """Heuristic: reject obvious HTML / markup blobs masquerading as excerpts."""
    s = text.strip()
    if not s:
        return True
    lowered = s.lower()
    if "<html" in lowered or "<body" in lowered or "<head" in lowered:
        return True
    if "</" in s and ">" in s:
        return True
    return bool(re.search(r"<\s*[a-zA-Z][^>]{0,200}>", s))


def validate_trusted_source_hit(hit: TrustedSourceHit) -> str | None:
    """
    Return None if ``hit`` is acceptable; otherwise a short machine reason string.
    """
    if excerpt_looks_like_html(hit.excerpt):
        return "excerpt_resembles_html"
    if not hit.source_key.strip():
        return "missing_source_key"
    return None


def partition_urls_by_allowlist(
    urls: list[HttpUrl],
    registry: TrustedSourceRegistryPort,
) -> tuple[list[TrustedRetrievalTarget], list[BlockedUrl]]:
    """
    Split ``urls`` into allowlisted targets (with ``source_key``) and blocked rows.

    Denylist (``TRUSTED_EXTERNAL_DENYLIST``) is enforced first, then ``registry.is_url_allowed``.
    No network I/O.
    """
    allowed: list[TrustedRetrievalTarget] = []
    blocked: list[BlockedUrl] = []
    seen: set[str] = set()
    denylist = get_trusted_url_denylist()

    for raw in urls:
        key = str(raw)
        if key in seen:
            continue
        seen.add(key)

        if not url_has_allowed_retrieval_scheme(raw):
            blocked.append(BlockedUrl(url=key, reason="unsupported_scheme"))
            continue

        if denylist.is_blocked(raw):
            blocked.append(BlockedUrl(url=key, reason="url_denied_by_policy"))
            continue

        if not registry.is_url_allowed(raw):
            blocked.append(BlockedUrl(url=key, reason="domain_not_allowlisted"))
            continue

        meta = registry.get_source_metadata(raw)
        if meta is None:
            blocked.append(BlockedUrl(url=key, reason="metadata_unresolved"))
            continue

        allowed.append(TrustedRetrievalTarget(url=raw, source_key=meta.source_key))

    return allowed, blocked


def sanitize_provider_hits(
    hits: list[TrustedSourceHit],
    *,
    registry: TrustedSourceRegistryPort,
) -> tuple[list[TrustedSourceHit], list[BlockedUrl]]:
    """
    Drop hits that are denylisted, fail excerpt rules, or are no longer allowlisted (defense in depth).

    Returns (kept_hits, synthetic_blocked_rows describing dropped hits).
    """
    kept: list[TrustedSourceHit] = []
    dropped: list[BlockedUrl] = []
    denylist = get_trusted_url_denylist()
    for hit in hits:
        if denylist.is_blocked(hit.url):
            dropped.append(BlockedUrl(url=str(hit.url), reason="post_provider_url_denied_by_policy"))
            continue
        if not registry.is_url_allowed(hit.url):
            dropped.append(BlockedUrl(url=str(hit.url), reason="post_provider_not_allowlisted"))
            continue
        reason = validate_trusted_source_hit(hit)
        if reason is not None:
            dropped.append(BlockedUrl(url=str(hit.url), reason=f"hit_validation:{reason}"))
            continue
        kept.append(hit)
    return kept, dropped


@runtime_checkable
class ExternalRetrievalProvider(Protocol):
    """
    Pluggable backend (HTTP client, vendor search API, enterprise connector, etc.).

    Implementations must not perform unrestricted crawling; callers should pass only
    pre-validated targets via :class:`TrustedRetrievalProviderRequest`.
    """

    @property
    def provider_id(self) -> str:
        """Stable id for logging and metrics (e.g. ``stub``, ``http_client_v1``)."""

    async def retrieve(self, request: TrustedRetrievalProviderRequest) -> TrustedRetrievalResult:
        """Return structured hits for the given targets (may be empty)."""


class StubExternalRetrievalProvider:
    """
    Non-network provider for tests and local development.

    Emits plain-text excerpts only—never HTML blobs.
    """

    @property
    def provider_id(self) -> str:
        return "stub"

    async def retrieve(self, request: TrustedRetrievalProviderRequest) -> TrustedRetrievalResult:
        now = datetime.now(timezone.utc)
        hits: list[TrustedSourceHit] = []
        n = len(request.targets)
        for i, target in enumerate(request.targets):
            host = str(target.url.host) if target.url.host else "unknown-host"
            excerpt = (
                f"Stub retrieval for {host}: no live document body was fetched. "
                "Replace StubExternalRetrievalProvider with a real ExternalRetrievalProvider "
                "when wiring a search or HTTP backend."
            )
            # Deterministic pseudo-scores so Layer2 can rank/dedupe like Tavily order.
            score = max(0.0, 0.88 - (i * (0.5 / max(n, 1))))
            hits.append(
                TrustedSourceHit(
                    url=target.url,
                    title=f"Stub title ({host})",
                    excerpt=excerpt,
                    source_key=target.source_key,
                    retrieved_at=now,
                    relevance_score=score,
                )
            )
        return TrustedRetrievalResult(hits=hits, blocked=[], provider_id=self.provider_id)


# ---------------------------------------------------------------------------
# Tavily Search (domain-scoped or global when registry wildcard ``*``)
# ---------------------------------------------------------------------------

TavilySearchTopic = Literal["general", "news", "finance"]
TavilySearchDepth = Literal["advanced", "basic", "fast", "ultra-fast"]


class TavilySearchResultItem(BaseModel):
    """One element of Tavily's ``results`` array (extra keys ignored)."""

    url: str = Field(..., min_length=1, max_length=2000)
    title: str = Field(default="", max_length=500)
    content: str = Field(default="", max_length=32000)
    score: float | None = Field(default=None)

    model_config = ConfigDict(extra="ignore")


class TavilySearchAPIResponse(BaseModel):
    """Typed subset of Tavily POST ``/search`` JSON used for mapping to :class:`TrustedSourceHit`."""

    query: str = ""
    results: list[TavilySearchResultItem] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


def _collapse_ws(text: str) -> str:
    return " ".join(text.replace("\r\n", "\n").replace("\r", "\n").split()).strip()


def _pick_tavily_query(context_queries: tuple[str, ...]) -> str:
    for raw in context_queries:
        s = _collapse_ws(raw)
        if s:
            return s[:2000]
    return "Trusted allowlisted reference search."


def _include_domains_from_targets(targets: tuple[TrustedRetrievalTarget, ...]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in targets:
        host = str(t.url.host).lower() if t.url.host else ""
        if not host:
            continue
        if host.startswith("www."):
            host = host[4:]
        if host not in seen:
            seen.add(host)
            out.append(host)
    return out


def _registry_allows_any_domain(registry: TrustedSourceRegistryPort) -> bool:
    """True when ``TRUSTED_EXTERNAL_ALLOWLIST_DOMAINS`` was ``*`` (wildcard registry)."""
    fn = getattr(registry, "allows_any_domain", None)
    return callable(fn) and fn() is True


def _excerpt_for_hit(item: TavilySearchResultItem) -> str:
    body = _collapse_ws(item.content)
    title = _collapse_ws(item.title)
    if len(body) < 16 and title:
        merged = _collapse_ws(f"{title}. {item.content}")
    else:
        merged = body
    if not merged:
        merged = title or "No excerpt returned for this result."
    return merged[:8000]


class TavilySearchProvider:
    """
    Provider-backed trusted search via Tavily's Search API only.

    With a normal registry, requests send ``include_domains`` derived from allowlisted ``targets``.
    With wildcard registry (env ``*``), ``include_domains`` is omitted so Tavily can return any host;
    post-filtering still uses ``registry.is_url_allowed`` / ``sanitize_provider_hits``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        registry: TrustedSourceRegistryPort,
        max_results: int = 5,
        search_depth: TavilySearchDepth = "basic",
        topic: TavilySearchTopic = "general",
        base_url: str = "https://api.tavily.com",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key.strip()
        self._registry = registry
        self._max_results = max(1, min(20, max_results))
        self._search_depth: TavilySearchDepth = search_depth
        self._topic: TavilySearchTopic = topic
        self._base_url = base_url.rstrip("/")
        self._http_client = http_client
        self._owns_client = http_client is None

    @property
    def provider_id(self) -> str:
        return "tavily"

    async def retrieve(self, request: TrustedRetrievalProviderRequest) -> TrustedRetrievalResult:
        now = datetime.now(timezone.utc)
        allow_any = _registry_allows_any_domain(self._registry)
        include_domains = _include_domains_from_targets(request.targets)
        if not self._api_key:
            return TrustedRetrievalResult(hits=[], blocked=[], provider_id=self.provider_id)
        if not allow_any and not include_domains:
            return TrustedRetrievalResult(hits=[], blocked=[], provider_id=self.provider_id)

        query = _pick_tavily_query(request.context_queries)
        logger.info(
            "tavily_search_invoked",
            extra={
                "event": "tavily_search",
                "include_domain_count": len(include_domains),
                "include_domains_omitted_for_global_search": allow_any,
                "max_results": self._max_results,
                "search_depth": self._search_depth,
                "query_char_len": len(query),
            },
        )
        url_adapter = TypeAdapter(HttpUrl)
        payload: dict[str, object] = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": self._search_depth,
            "max_results": self._max_results,
            "include_answer": False,
            "topic": self._topic,
        }
        if not allow_any:
            payload["include_domains"] = include_domains

        client = self._http_client or httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        blocked: list[BlockedUrl] = []
        hits: list[TrustedSourceHit] = []
        denylist = get_trusted_url_denylist()
        try:
            resp = await client.post(f"{self._base_url}/search", json=payload)
            if resp.status_code >= 400:
                blocked.append(
                    BlockedUrl(
                        url="https://api.tavily.com/search",
                        reason=f"tavily_http_{resp.status_code}",
                    )
                )
                return TrustedRetrievalResult(hits=[], blocked=blocked, provider_id=self.provider_id)

            parsed = TavilySearchAPIResponse.model_validate(resp.json())
            for item in parsed.results:
                try:
                    page_url = url_adapter.validate_python(item.url)
                except Exception:
                    blocked.append(BlockedUrl(url=item.url[:2000], reason="tavily_result_url_invalid"))
                    continue

                if denylist.is_blocked(page_url):
                    blocked.append(
                        BlockedUrl(url=str(page_url), reason="tavily_result_url_denied_by_policy")
                    )
                    continue

                if not self._registry.is_url_allowed(page_url):
                    blocked.append(
                        BlockedUrl(url=str(page_url), reason="tavily_result_domain_not_allowlisted")
                    )
                    continue

                meta = self._registry.get_source_metadata(page_url)
                if meta is None:
                    blocked.append(
                        BlockedUrl(url=str(page_url), reason="tavily_result_metadata_unresolved")
                    )
                    continue

                excerpt = _excerpt_for_hit(item)
                if excerpt_looks_like_html(excerpt):
                    blocked.append(
                        BlockedUrl(url=str(page_url), reason="tavily_excerpt_resembles_html")
                    )
                    continue

                title_out = _collapse_ws(item.title)[:500] or None
                rel = float(item.score) if item.score is not None else None
                hits.append(
                    TrustedSourceHit(
                        url=page_url,
                        title=title_out,
                        excerpt=excerpt,
                        source_key=meta.source_key,
                        retrieved_at=now,
                        relevance_score=rel,
                    )
                )
        except httpx.RequestError as exc:
            logger.warning("Tavily search request failed", extra={"error": str(exc)})
            blocked.append(BlockedUrl(url="https://api.tavily.com/search", reason="tavily_transport_error"))
        finally:
            if self._owns_client:
                await client.aclose()

        return TrustedRetrievalResult(hits=hits, blocked=blocked, provider_id=self.provider_id)


def build_external_retrieval_provider(*, registry: TrustedSourceRegistryPort) -> ExternalRetrievalProvider:
    """
    Select the configured :class:`ExternalRetrievalProvider`.

    Uses :class:`StubExternalRetrievalProvider` when trusted external retrieval is disabled,
    when ``TRUSTED_SEARCH_PROVIDER=stub``, when Tavily is selected but ``TAVILY_API_KEY`` is empty,
    or on configuration errors (logged).
    """
    from src.core.config import settings

    if not settings.enable_trusted_external_retrieval:
        return StubExternalRetrievalProvider()

    provider_name = settings.trusted_search_provider.strip().lower()
    if provider_name == "tavily":
        key = settings.tavily_api_key.strip()
        if not key:
            logger.warning(
                "TRUSTED_SEARCH_PROVIDER=tavily but TAVILY_API_KEY is empty; using stub retrieval provider"
            )
            return StubExternalRetrievalProvider()
        return TavilySearchProvider(
            api_key=key,
            registry=registry,
            max_results=settings.tavily_max_results,
            search_depth=cast(TavilySearchDepth, settings.tavily_search_depth),
            topic=cast(TavilySearchTopic, settings.tavily_topic),
            base_url=settings.tavily_api_base_url,
        )

    return StubExternalRetrievalProvider()


class TrustedExternalRetrievalService:
    """
    Orchestrates allowlist enforcement, provider dispatch, and post-provider validation.

    This is the supported entry point for trusted external retrieval in the app flow.
    """

    def __init__(
        self,
        *,
        registry: TrustedSourceRegistryPort,
        provider: ExternalRetrievalProvider | None = None,
    ) -> None:
        self._registry = registry
        self._provider = provider or StubExternalRetrievalProvider()

    async def retrieve(self, retrieval_input: TrustedRetrievalInput) -> TrustedRetrievalResult:
        """
        Validate ``candidate_urls`` against the registry, then delegate to the provider.

        ``context_queries`` are forwarded for future search providers; the stub ignores them.
        """
        allowed, blocked = partition_urls_by_allowlist(retrieval_input.candidate_urls, self._registry)
        provider_request = TrustedRetrievalProviderRequest(
            targets=tuple(allowed),
            context_queries=tuple(retrieval_input.context_queries),
        )
        raw = await self._provider.retrieve(provider_request)

        if raw.provider_id != self._provider.provider_id:
            logger.warning(
                "Provider result provider_id mismatch",
                extra={"expected": self._provider.provider_id, "got": raw.provider_id},
            )

        kept, dropped = sanitize_provider_hits(raw.hits, registry=self._registry)
        merged_blocked = [*blocked, *raw.blocked, *dropped]
        return TrustedRetrievalResult(
            hits=kept,
            blocked=merged_blocked,
            provider_id=self._provider.provider_id,
        )


# ---------------------------------------------------------------------------
# Single-URL fetch (legacy / fine-grained pipelines)
# ---------------------------------------------------------------------------


class FetchedPage(BaseModel):
    """Raw fetch result for one URL (transport layer; not the primary structured hit)."""

    url: HttpUrl
    source_key: str = Field(..., min_length=1, max_length=64)
    page_title: str | None = Field(
        default=None,
        max_length=500,
        description="Document title from scrape metadata when the provider exposes it.",
    )
    http_status: int | None = Field(default=None, ge=100, le=599)
    content_type: str | None = Field(default=None, max_length=128)
    body_text: str | None = Field(
        default=None,
        description="Plain text or lightly normalized body; HTML parsing is upstream",
    )
    retrieved_at: datetime
    fetch_error: str | None = Field(default=None, max_length=2000)

    model_config = ConfigDict(extra="forbid")


@runtime_checkable
class WebRetriever(Protocol):
    """Fetch a single page given a URL that has already passed allowlist checks."""

    async def fetch_page(self, *, url: HttpUrl, source_key: str) -> FetchedPage:
        """
        Retrieve one URL and return structured fetch metadata plus body text when available.

        On transport or HTTP errors, populate fetch_error and optional http_status without raising
        unless the contract explicitly requires fail-fast (implementations may choose).
        """


class NullWebRetriever:
    """
    Placeholder retriever that performs no network I/O.

    Returns an empty body and a note in fetch_error for tracing in tests.
    """

    async def fetch_page(self, *, url: HttpUrl, source_key: str) -> FetchedPage:
        """No-op fetch: documents that low-level HTTP wiring is not yet active."""
        return FetchedPage(
            url=url,
            source_key=source_key,
            http_status=None,
            content_type=None,
            body_text=None,
            retrieved_at=datetime.now(timezone.utc),
            fetch_error="NullWebRetriever: network fetch not implemented",
        )
