"""
Firecrawl-backed fetch for allowlisted URLs not on the denylist (no open crawling).

Uses Firecrawl ``POST /v2/scrape`` with ``formats: [\"markdown\"]`` and returns
:class:`FetchedPage` rows suitable for :func:`src.research.evidence_extractor._trusted_hit_from_fetched_page`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import HttpUrl

from src.research.source_registry import TrustedSourceRegistryPort
from src.research.url_denylist import get_trusted_url_denylist
from src.research.web_retriever import FetchedPage
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _dedupe_and_cap_pages(
    pages: Sequence[tuple[HttpUrl, str]],
    *,
    max_urls: int,
) -> list[tuple[HttpUrl, str]]:
    out: list[tuple[HttpUrl, str]] = []
    seen: set[str] = set()
    for url, source_key in pages:
        key = str(url)
        if key in seen:
            continue
        seen.add(key)
        out.append((url, source_key))
        if len(out) >= max_urls:
            break
    return out


def _firecrawl_timeout_ms(timeout_s: int) -> int:
    ms = int(timeout_s * 1000)
    return max(1000, min(ms, 300_000))


class FirecrawlContentProvider:
    """
    Provider-backed extraction: one HTTP call per URL to Firecrawl scrape (markdown).

    Callers must pass URLs already tied to an allowlist policy; each request is re-checked with
    ``registry.is_url_allowed`` before leaving the process.
    """

    def __init__(
        self,
        *,
        api_key: str,
        api_base_url: str = "https://api.firecrawl.dev",
        max_urls_per_request: int = 3,
        timeout_s: int = 30,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key.strip()
        self._base = api_base_url.strip().rstrip("/")
        self._max_urls_per_request = max(0, min(20, max_urls_per_request))
        self._timeout_s = max(5, min(180, timeout_s))
        self._http_client = http_client
        self._owns_client = http_client is None

    @property
    def max_urls_per_request(self) -> int:
        return self._max_urls_per_request

    async def fetch_trusted_pages(
        self,
        pages: Sequence[tuple[HttpUrl, str]],
        *,
        registry: TrustedSourceRegistryPort,
    ) -> list[FetchedPage]:
        """
        Fetch markdown (or structured JSON body) for up to ``max_urls_per_request`` unique URLs.

        Returns one :class:`FetchedPage` per input tuple after dedupe+cap (skipped rows are omitted).
        """
        now = datetime.now(timezone.utc)
        if self._max_urls_per_request == 0 or not self._api_key:
            return []

        work = _dedupe_and_cap_pages(pages, max_urls=self._max_urls_per_request)
        if not work:
            return []

        denylist = get_trusted_url_denylist()

        logger.info(
            "firecrawl_batch_invoked",
            extra={
                "event": "firecrawl_scrape",
                "url_count": len(work),
            },
        )

        client = self._http_client or httpx.AsyncClient(timeout=httpx.Timeout(self._timeout_s + 5.0))
        out: list[FetchedPage] = []
        try:
            for url, source_key in work:
                if denylist.is_blocked(url):
                    out.append(
                        FetchedPage(
                            url=url,
                            source_key=source_key,
                            http_status=None,
                            content_type=None,
                            body_text=None,
                            retrieved_at=now,
                            fetch_error="url_denied_by_policy",
                        )
                    )
                    continue
                if not registry.is_url_allowed(url):
                    out.append(
                        FetchedPage(
                            url=url,
                            source_key=source_key,
                            http_status=None,
                            content_type=None,
                            body_text=None,
                            retrieved_at=now,
                            fetch_error="domain_not_allowlisted",
                        )
                    )
                    continue
                page = await self._scrape_one(client=client, url=url, source_key=source_key, now=now)
                out.append(page)
        finally:
            if self._owns_client:
                await client.aclose()
        return out

    async def _scrape_one(
        self,
        *,
        client: httpx.AsyncClient,
        url: HttpUrl,
        source_key: str,
        now: datetime,
    ) -> FetchedPage:
        scrape_url = f"{self._base}/v2/scrape"
        payload: dict[str, Any] = {
            "url": str(url),
            "formats": ["markdown"],
            "onlyMainContent": True,
            "timeout": _firecrawl_timeout_ms(self._timeout_s),
        }
        try:
            resp = await client.post(
                scrape_url,
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except httpx.RequestError as exc:
            logger.warning("Firecrawl scrape transport error", extra={"url": str(url), "error": str(exc)})
            return FetchedPage(
                url=url,
                source_key=source_key,
                http_status=None,
                content_type=None,
                body_text=None,
                retrieved_at=now,
                fetch_error=f"firecrawl_transport_error:{exc!s}"[:2000],
            )

        body_text: str | None = None
        content_type = "text/markdown"
        fetch_error: str | None = None
        http_status = resp.status_code

        try:
            data = resp.json()
        except json.JSONDecodeError:
            return FetchedPage(
                url=url,
                source_key=source_key,
                http_status=http_status,
                content_type=None,
                body_text=None,
                retrieved_at=now,
                fetch_error="firecrawl_invalid_json_response",
            )

        if http_status >= 400:
            err = data.get("error") if isinstance(data, dict) else None
            msg = str(err or resp.text or f"http_{http_status}")[:2000]
            return FetchedPage(
                url=url,
                source_key=source_key,
                http_status=http_status,
                content_type=None,
                body_text=None,
                retrieved_at=now,
                fetch_error=msg,
            )

        if not isinstance(data, dict):
            return FetchedPage(
                url=url,
                source_key=source_key,
                http_status=http_status,
                content_type=None,
                body_text=None,
                retrieved_at=now,
                fetch_error="firecrawl_unexpected_response_shape",
            )

        if data.get("success") is False:
            err = data.get("error") or data.get("message") or "firecrawl_success_false"
            return FetchedPage(
                url=url,
                source_key=source_key,
                http_status=http_status,
                content_type=None,
                body_text=None,
                retrieved_at=now,
                fetch_error=str(err)[:2000],
            )

        inner = data.get("data")
        if not isinstance(inner, dict):
            inner = {}

        page_title: str | None = None
        raw_title = inner.get("title")
        if isinstance(raw_title, str) and raw_title.strip():
            page_title = raw_title.strip()[:500]
        else:
            meta = inner.get("metadata")
            if isinstance(meta, dict):
                mt = meta.get("title")
                if isinstance(mt, str) and mt.strip():
                    page_title = mt.strip()[:500]

        md = inner.get("markdown")
        if isinstance(md, str) and md.strip():
            body_text = md.strip()[:500_000]
        else:
            # Optional fallback: JSON-like string payloads from structured extract (not used by default)
            raw_json = inner.get("json") or inner.get("content")
            if isinstance(raw_json, (dict, list)):
                body_text = json.dumps(raw_json, ensure_ascii=False, indent=2)[:500_000]
                content_type = "application/json"
            elif isinstance(raw_json, str) and raw_json.strip():
                body_text = raw_json.strip()[:500_000]

        if not body_text:
            html = inner.get("html")
            if isinstance(html, str) and html.strip():
                body_text = html.strip()[:500_000]
                content_type = "text/html"

        if not body_text:
            fetch_error = "firecrawl_empty_extracted_body"

        return FetchedPage(
            url=url,
            source_key=source_key,
            page_title=page_title,
            http_status=http_status,
            content_type=content_type if body_text else None,
            body_text=body_text,
            retrieved_at=now,
            fetch_error=fetch_error,
        )


def build_trusted_content_fetcher() -> FirecrawlContentProvider | None:
    """
    Instantiate :class:`FirecrawlContentProvider` when flags and env allow; otherwise ``None``.

    ``None`` means callers keep search-provider excerpts without a second fetch step.
    """
    from src.core.config import settings

    if not settings.enable_trusted_external_retrieval:
        return None
    if settings.trusted_content_provider.strip().lower() != "firecrawl":
        return None
    key = settings.firecrawl_api_key.strip()
    if not key:
        logger.warning(
            "TRUSTED_CONTENT_PROVIDER=firecrawl but FIRECRAWL_API_KEY is empty; skipping Firecrawl fetch"
        )
        return None
    return FirecrawlContentProvider(
        api_key=key,
        api_base_url=settings.firecrawl_api_base_url,
        max_urls_per_request=settings.firecrawl_max_urls_per_request,
        timeout_s=settings.firecrawl_timeout_seconds,
    )
