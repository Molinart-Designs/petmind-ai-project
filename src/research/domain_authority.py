"""
Domain-tier authority caps for trusted external evidence (Layer 2).

Used for source ``authority_score``, ranking, confidence, and persistence gates.
Registry-configured scores are clamped so high-risk / anecdotal hosts never look “trusted”.
"""

from __future__ import annotations

from src.research.http_url_utils import domain_from_http_url


def _norm_host(host: str) -> str:
    h = (host or "").strip().lower().rstrip(".")
    if h.startswith("www."):
        h = h[4:]
    return h


def is_anecdotal_social_domain(host: str) -> bool:
    """Forums / UGC where answers are anecdotal, not clinical references."""
    h = _norm_host(host)
    return h == "reddit.com" or h.endswith(".reddit.com")


def is_reddit_supplemental_domain(host: str) -> bool:
    """reddit.com must never be treated as primary trusted evidence — supplemental only."""
    return is_anecdotal_social_domain(host)


# Minimum clamped authority for a non-Reddit snippet to count as an anchor when Reddit is present.
ANCHOR_AUTHORITY_THRESHOLD_FOR_REDDIT_MIX: float = 0.38


def snippet_is_supplemental_reddit(sn: object) -> bool:
    """True when the snippet URL host is Reddit (anecdotal supplemental tier)."""
    url = getattr(sn, "evidence_page_url", None) or ""
    return is_reddit_supplemental_domain(domain_from_http_url(str(url).strip()))


def snippet_is_anchor_trusted_evidence(sn: object) -> bool:
    """
    Higher-authority anchor: not Reddit and at/above anchor authority threshold.

    Used so Reddit may appear in a bundle only alongside at least one such snippet (L3 persistence
    and merged context selection).
    """
    url = getattr(sn, "evidence_page_url", None) or ""
    host = domain_from_http_url(str(url).strip())
    if is_anecdotal_social_domain(host):
        return False
    try:
        auth = float(getattr(sn, "authority_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    return auth >= ANCHOR_AUTHORITY_THRESHOLD_FOR_REDDIT_MIX


def is_pitpat_domain(host: str) -> bool:
    h = _norm_host(host)
    return h == "pitpat.com" or h.endswith(".pitpat.com")


def domain_authority_score(host: str) -> float:
    """
    Baseline authority weight for snippets tied to this host (0–1).

    - reddit: low (anecdotal UGC)
    - pitpat: medium (consumer pet-care product guidance)
    - other allowlisted: neutral provisional default
    """
    h = _norm_host(host)
    if not h:
        return 0.35
    if is_anecdotal_social_domain(h):
        return 0.24
    if is_pitpat_domain(h):
        return 0.56
    return 0.48


def domain_authority_cap(host: str) -> float:
    """
    Hard ceiling on authority for this host when blending with registry scores.

    Reddit never exceeds ~0.30 even if misconfigured in the registry.
    """
    h = _norm_host(host)
    if not h:
        return 0.45
    if is_anecdotal_social_domain(h):
        return 0.30
    if is_pitpat_domain(h):
        return 0.68
    return 0.98


def clamp_registry_authority(host: str, registry_authority: float) -> float:
    """Combine registry intent with domain safety caps."""
    cap = domain_authority_cap(host)
    base = domain_authority_score(host)
    blended = 0.5 * registry_authority + 0.5 * base
    return float(min(cap, max(0.0, min(1.0, blended))))


def clamp_synthetic_authority(host: str) -> float:
    """Authority for per-hit synthetic sources (no registry row)."""
    return float(min(domain_authority_cap(host), max(0.0, min(1.0, domain_authority_score(host)))))
