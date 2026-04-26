"""
Heuristics for external snippet quality, deduplication, scope hints, and ranking signals.

Used by :mod:`src.research.evidence_extractor` and :mod:`src.security.guardrails` (confidence).
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from src.research.external_ranking import token_overlap_score

_MARKDOWN_IMG = re.compile(r"!\[[^\]\n]{0,800}\]\([^)\n]{1,2000}\)")
_LINE_IMG_ONLY = re.compile(
    r"^\s*(\[image[^\]]*\]|image:\s*|!\[[^\]]*\]\([^)]+\))\s*$",
    re.I,
)
_HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_QUESTIONISH = re.compile(r"\?\s*$")

_PUPPY_TERMS = re.compile(
    r"\b(pupp(y|ies)|neonatal|weaning|first\s+vaccination\s+series|8\s*weeks?\s+old)\b",
    re.I,
)
_ADULT_TERMS_Q = re.compile(
    r"\b(adult\s+dog|mature\s+dog|my\s+dog\s+is\s+\d+|years?\s+old|senior\s+dog|grown\s+dog)\b",
    re.I,
)

_BREED_ALIASES: tuple[tuple[str, str], ...] = (
    ("pomeranian", "pomeranian"),
    ("pom\b", "pomeranian"),
    ("labrador", "labrador"),
    ("golden retriever", "golden retriever"),
    ("german shepherd", "german shepherd"),
    ("border collie", "border collie"),
    ("french bulldog", "french bulldog"),
    ("beagle", "beagle"),
)


def normalize_article_url_for_dedupe(url: str) -> str:
    """Stable key for per-article public source deduplication."""
    s = (url or "").strip()
    if not s:
        return ""
    if "://" not in s:
        s = f"https://{s}"
    p = urlparse(s)
    host = (p.hostname or "").lower()
    path = p.path or ""
    return f"{host}{path}".rstrip("/")


def dedupe_similar_claim_units(units: list[str], *, prefix_chars: int = 88) -> list[str]:
    """Drop units that repeat the same opening (common duplicated headings on one page)."""
    seen: set[str] = set()
    out: list[str] = []
    for u in units:
        key = " ".join(u.split())[:prefix_chars].lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(u)
    return out


def strip_inline_markdown_images(text: str) -> str:
    t = _MARKDOWN_IMG.sub(" ", text)
    return re.sub(r"\s{2,}", " ", t).strip()


def snippet_is_markdown_image_heavy(text: str) -> bool:
    raw = text.strip()
    if not raw:
        return True
    without = strip_inline_markdown_images(raw)
    if len(without) < 50 and _MARKDOWN_IMG.search(raw):
        return True
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if lines and all(_LINE_IMG_ONLY.match(ln) for ln in lines):
        return True
    return False


def snippet_resembles_heading_or_title_line(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return True
    first = lines[0]
    if _HEADING_LINE.match(first):
        rest = "\n".join(lines[1:]).strip()
        # Page excerpt that starts with a markdown title but continues with prose is not "heading-only".
        if len(rest) >= 50:
            return False
        if len(t) < 110 and len(lines) <= 2:
            return True
        return len(rest) < 40
    if len(t) < 90 and not re.search(r"\d", t) and t.count(".") <= 1 and len(t.split()) <= 14:
        if t.isupper() and len(t) > 10:
            return True
    return False


def snippet_repeats_question_heading(text: str, question: str) -> bool:
    q = (question or "").strip().lower()
    t = " ".join(text.split()).strip().lower()
    if len(t) < 160 and q and len(q) >= 12:
        ov = token_overlap_score(q, t)
        if ov >= 0.58 and len(t.split()) <= 22:
            return True
        if t in q or (len(t) >= 10 and t == q[: len(t)]):
            return True
    if _QUESTIONISH.search(t) and len(t.split()) <= 16:
        ov = token_overlap_score(q, t)
        if ov >= 0.45:
            return True
    return False


def snippet_is_generic_nav_or_decorative(text: str) -> bool:
    low = text.lower().strip()
    if len(low) < 70 and any(
        x in low
        for x in (
            "read more",
            "click here",
            "related articles",
            "you may also like",
            "table of contents",
            "share this",
        )
    ):
        return True
    return False


def snippet_puppy_mismatch_for_adult_dog_question(text: str, question: str) -> bool:
    """True when the question clearly concerns an adult dog but the snippet is puppy-centric."""
    q = (question or "").lower()
    t = (text or "").lower()
    if not _ADULT_TERMS_Q.search(q) and "adult" not in q and not re.search(r"\d+\s*years?\s*old", q):
        return False
    if "puppy" not in q and "puppies" not in q:
        if _PUPPY_TERMS.search(t) and not re.search(r"\badult\b|\bsenior\b|\bmature\b", t):
            pup_hits = len(_PUPPY_TERMS.findall(t))
            if pup_hits >= 2 or (pup_hits >= 1 and len(t) < 220):
                return True
    return False


def should_discard_snippet_unit(text: str, question: str, *, page_title: str | None = None) -> bool:
    """Filter low-value units before persistence / LLM context."""
    t = (text or "").strip()
    if not t:
        return True
    if snippet_is_markdown_image_heavy(t):
        return True
    if snippet_resembles_heading_or_title_line(t) and len(t) < 120:
        return True
    if snippet_repeats_question_heading(t, question):
        return True
    if snippet_is_generic_nav_or_decorative(t):
        return True
    if snippet_puppy_mismatch_for_adult_dog_question(t, question):
        return True
    title = (page_title or "").strip()
    if title and len(t) < 140:
        if token_overlap_score(title.lower(), t.lower()) >= 0.72:
            return True
    return False


def direct_answer_signal(text: str, question: str) -> float:
    """Higher when the unit looks like explanatory guidance, not a heading."""
    t = (text or "").strip()
    q = (question or "").strip()
    if len(t) < 70:
        return 0.0
    ov = token_overlap_score(q, t)
    instruct = bool(
        re.search(
            r"\b(offer|provide|ensure|monitor|measure|feed|avoid|consult|schedule|limit|increase|reduce)\b",
            t,
            re.I,
        )
    )
    factual = bool(re.search(r"\b\d{1,3}\s*(mg|ml|g|hours?|days?|times|percent|%)\b", t, re.I))
    score = 0.35 * min(1.0, ov / 0.35) + (0.35 if instruct else 0.0) + (0.3 if factual else 0.0)
    if snippet_resembles_heading_or_title_line(t):
        score *= 0.35
    return max(0.0, min(1.0, score))


def breed_specificity_signal(text: str, question: str, *, page_title: str | None = None) -> float:
    """Reward breed- or topic-specific overlap with the question."""
    q = (question or "").lower()
    blob = f"{(page_title or '').lower()} {(text or '').lower()}"
    score = 0.0
    for pat, _label in _BREED_ALIASES:
        if re.search(pat, q, re.I) and re.search(pat, blob, re.I):
            score = max(score, 1.0)
    if score == 0.0 and len(q) > 20:
        ov = token_overlap_score(q, blob[:800])
        if ov >= 0.22:
            score = 0.45
        if ov >= 0.32:
            score = 0.65
    return max(0.0, min(1.0, score))


def noise_signal(text: str, question: str, *, page_title: str | None = None) -> float:
    """0 = clean, toward 1 = noisy / weak for ranking penalty."""
    t = (text or "").strip()
    parts = [
        0.55 if snippet_is_markdown_image_heavy(t) else 0.0,
        0.4 if snippet_resembles_heading_or_title_line(t) else 0.0,
        0.45 if snippet_repeats_question_heading(t, question) else 0.0,
        0.35 if snippet_is_generic_nav_or_decorative(t) else 0.0,
    ]
    title = (page_title or "").strip()
    if title and len(t) < 160 and token_overlap_score(title.lower(), t.lower()) >= 0.65:
        parts.append(0.35)
    return max(0.0, min(1.0, max(parts)))


def infer_snippet_scope_fields(
    question: str,
    *,
    page_title: str | None,
    unit_text: str,
) -> dict[str, str]:
    """
    Conservative species/breed/life_stage when strongly supported by question + page/snippet text.

    Returns only keys that pass the bar (caller merges onto snippet).
    """
    q = (question or "").strip().lower()
    blob = f"{(page_title or '').lower()} {(unit_text or '').lower()}"
    out: dict[str, str] = {}

    if ("dog" in q or "canine" in q or "pomeranian" in q or "puppy" in q) and "cat" not in q:
        if "dog" in blob or "canine" in blob or "pomeranian" in blob:
            out["species"] = "dog"

    for pat, label in _BREED_ALIASES:
        if re.search(pat, q, re.I) and re.search(pat, blob, re.I):
            out["breed"] = label
            break

    if re.search(r"\badult\s+dog\b|\badult\b.*\bdog\b|\badult\b.*\bpomeranian\b|\d+\s*years?\s*old", q, re.I):
        if re.search(r"\badult\b|\bmature\b|\bsenior\b", blob, re.I) or re.search(
            r"\d+\s*years?\s*old", q, re.I
        ):
            out["life_stage"] = "adult"
    if "senior dog" in q or "elderly dog" in q:
        if "senior" in blob or "older" in blob:
            out["life_stage"] = "senior"

    return out


def highly_relevant_direct_answer_chunk(
    *,
    snippet_text: str,
    similarity_score: float,
    question: str,
) -> bool:
    """Used by guardrails for confidence calibration on provisional external chunks."""
    t = (snippet_text or "").strip()
    if not t or should_discard_snippet_unit(t, question, page_title=None):
        return False
    ov = token_overlap_score(question, t)
    rank = float(similarity_score)
    return ov >= 0.17 and rank >= 0.56 and direct_answer_signal(t, question) >= 0.35
