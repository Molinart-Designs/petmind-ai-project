"""Small URL helpers shared by evidence extraction and envelopes (no heavy imports)."""

from __future__ import annotations

from urllib.parse import urlparse


def domain_from_http_url(url: str) -> str:
    """Return lowercase hostname from an HTTP(S) URL, or empty string if missing."""
    if not url or not isinstance(url, str):
        return ""
    parsed = urlparse(url.strip())
    host = parsed.hostname
    return (host or "").lower()
