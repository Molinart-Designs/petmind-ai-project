"""
Explicit denylist for trusted external URLs (Layer 2).

Rules apply **after** scheme checks and **before** allowlist matching: a URL on the denylist is
never retrieved, even if its host appears on ``TRUSTED_EXTERNAL_ALLOWLIST_DOMAINS`` (or wildcard ``*``).
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from pydantic import HttpUrl

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _norm_host(host: str) -> str:
    h = (host or "").strip().lower().rstrip(".")
    if h.startswith("www."):
        h = h[4:]
    return h


def _host_matches_base(url_host: str, base_host: str) -> bool:
    """True when ``url_host`` equals ``base_host`` or is a subdomain of it."""
    bh = _norm_host(base_host)
    uh = _norm_host(url_host)
    if not bh or not uh:
        return False
    return uh == bh or uh.endswith(f".{bh}")


def _url_host_and_path(url: HttpUrl | str) -> tuple[str | None, str]:
    s = str(url).strip()
    if not s:
        return None, "/"
    if "://" not in s:
        s = f"https://{s}"
    parsed = urlparse(s)
    host = _norm_host(parsed.hostname or "")
    path = (parsed.path or "").strip()
    if not path:
        path = "/"
    return host, path


def _path_matches_prefix(url_path: str, prefix: str) -> bool:
    if url_path == prefix:
        return True
    if not prefix.endswith("/"):
        return url_path.startswith(prefix + "/")
    return url_path.startswith(prefix)


@dataclass(frozen=True)
class _HostDenyRule:
    host: str


@dataclass(frozen=True)
class _PathDenyRule:
    host: str
    path_prefix: str


class TrustedUrlDenylist:
    """Parsed ``TRUSTED_EXTERNAL_DENYLIST`` — thread-safe, immutable."""

    __slots__ = ("_rules",)

    def __init__(self, rules: tuple[_HostDenyRule | _PathDenyRule, ...]) -> None:
        self._rules = rules

    @classmethod
    def from_raw(cls, raw: str) -> TrustedUrlDenylist:
        rules: list[_HostDenyRule | _PathDenyRule] = []
        for part in (raw or "").split(","):
            seg = part.strip()
            if not seg:
                continue
            parsed = _parse_segment(seg)
            if parsed is not None:
                rules.append(parsed)
            else:
                logger.warning(
                    "Skipping invalid trusted external denylist segment",
                    extra={"segment_preview": seg[:160]},
                )
        return cls(tuple(rules))

    def is_blocked(self, url: HttpUrl | str) -> bool:
        if not self._rules:
            return False
        uh, upath = _url_host_and_path(url)
        if not uh:
            return False
        for rule in self._rules:
            if isinstance(rule, _HostDenyRule):
                if _host_matches_base(uh, rule.host):
                    return True
            elif isinstance(rule, _PathDenyRule):
                if _host_matches_base(uh, rule.host) and _path_matches_prefix(upath, rule.path_prefix):
                    return True
        return False


def _parse_segment(seg: str) -> _HostDenyRule | _PathDenyRule | None:
    if "://" in seg:
        parsed = urlparse(seg)
        if parsed.scheme not in ("http", "https"):
            return None
        host = _norm_host(parsed.hostname or "")
        if not host or host == "*":
            return None
        path = parsed.path or ""
        if path in ("", "/"):
            return _HostDenyRule(host=host)
        path_norm = path if path.startswith("/") else f"/{path}"
        while len(path_norm) > 1 and path_norm.endswith("/"):
            path_norm = path_norm.rstrip("/")
        return _PathDenyRule(host=host, path_prefix=path_norm)

    if " " in seg or ".." in seg:
        return None

    if "/" in seg:
        host_part, rest = seg.split("/", 1)
        host = _norm_host(host_part)
        if not host or host == "*":
            return None
        rest_stripped = rest.strip().strip("/")
        if not rest_stripped:
            return _HostDenyRule(host=host)
        path_norm = "/" + rest_stripped
        return _PathDenyRule(host=host, path_prefix=path_norm)

    host = _norm_host(seg)
    if not host or host == "*":
        return None
    return _HostDenyRule(host=host)


def get_trusted_url_denylist() -> TrustedUrlDenylist:
    """Build denylist from current process settings (no I/O)."""
    from src.core.config import settings

    raw = (getattr(settings, "trusted_external_denylist", None) or "").strip()
    return TrustedUrlDenylist.from_raw(raw)
