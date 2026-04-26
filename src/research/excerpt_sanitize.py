"""
Strip common navigation / layout junk from web excerpts before evidence extraction.
"""

from __future__ import annotations

import re

# Avoid DOTALL: a greedy `[^)]` span could swallow newlines and erase whole excerpts by mistake.
_MARKDOWN_IMG = re.compile(r"!\[[^\]\n]{0,800}\]\([^)\n]{1,2000}\)")


def _strip_markdown_images(text: str) -> str:
    t = _MARKDOWN_IMG.sub(" ", text)
    return re.sub(r"\s{2,}", " ", t).strip()

_NAV_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*skip\s+to\s+(main\s+)?content\.?\s*$", re.I),
    re.compile(r"^\s*skip\s+to\s+footer\.?\s*$", re.I),
    re.compile(r"^\s*featured\s+image\.?\s*$", re.I),
    re.compile(r"^\s*cookie\s+(policy|consent|preferences).*$", re.I),
    re.compile(r"^\s*(accept|reject)\s+all\s+cookies\.?\s*$", re.I),
    re.compile(r"^\s*subscribe\s+to\s+our\s+newsletter\.?\s*$", re.I),
    re.compile(r"^\s*sign\s+up\s+for\s+updates\.?\s*$", re.I),
    re.compile(r"^\s*privacy\s+policy\.?\s*$", re.I),
    re.compile(r"^\s*terms\s+of\s+(use|service)\.?\s*$", re.I),
    re.compile(r"^\s*all\s+rights\s+reserved\.?\s*$", re.I),
    re.compile(r"^\s*copyright\s+©?\s*\d{4}.*$", re.I),
    re.compile(r"^\s*share\s+on\s+(facebook|twitter|x|linkedin)\s*$", re.I),
)

_NAV_PHRASES: tuple[str, ...] = (
    "skip to main content",
    "skip to content",
    "skip to navigation",
    "table of contents",
    "jump to content",
    "breadcrumb",
)


def remove_navigation_junk_lines(text: str) -> str:
    """Drop whole lines that look like chrome/navigation boilerplate."""
    out_lines: list[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = line.strip()
        if not stripped:
            out_lines.append("")
            continue
        low = stripped.lower()
        if any(p.match(stripped) for p in _NAV_LINE_PATTERNS):
            continue
        if any(phrase in low for phrase in _NAV_PHRASES) and len(stripped) < 120:
            continue
        out_lines.append(line.rstrip())
    joined = "\n".join(out_lines)
    return re.sub(r"\n{3,}", "\n\n", joined).strip()


def scrub_inline_nav_phrases(text: str) -> str:
    """Remove short inline boilerplate segments (conservative)."""
    t = text
    for phrase in _NAV_PHRASES:
        t = re.sub(re.escape(phrase), " ", t, flags=re.I)
    return re.sub(r"\s{2,}", " ", t).strip()


def clean_excerpt_for_evidence(text: str) -> str:
    """Full excerpt cleanup before splitting into claim units."""
    if not text or not text.strip():
        return ""
    # Remove navigation lines first so inline junk checks do not span merged paragraphs.
    t = remove_navigation_junk_lines(text)
    t = _strip_markdown_images(t)
    t = scrub_inline_nav_phrases(t)
    return t.strip()
