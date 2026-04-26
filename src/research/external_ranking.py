"""
Composite ranking for external snippets: provider relevance, domain authority, query overlap.
"""

from __future__ import annotations

import re


def normalize_provider_relevance(raw: float | None) -> float:
    """Map provider scores into ~[0,1] (Tavily is typically 0–1; guard other scales)."""
    if raw is None:
        return 0.38
    x = float(raw)
    if x > 1.0:
        if x <= 10.0:
            x = x / 10.0
        else:
            x = 1.0
    if x < 0.0:
        x = 0.0
    return min(1.0, x)


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Záéíóúñ]{3,}", text.lower())
    return {w for w in words if len(w) >= 3}


def token_overlap_score(question: str, snippet_text: str) -> float:
    """Cheap token-overlap score in [0,1] (not embeddings)."""
    if not question.strip() or not snippet_text.strip():
        return 0.0
    q = _tokenize(question)
    s = _tokenize(snippet_text)
    if not q or not s:
        return 0.0
    inter = len(q & s)
    union = len(q | s)
    return float(inter) / float(union) if union else 0.0


def composite_snippet_ranking(
    *,
    provider_relevance: float,
    source_authority: float,
    query_overlap: float,
    direct_answer_signal: float = 0.0,
    specificity_signal: float = 0.0,
    noise_signal: float = 0.0,
) -> float:
    """
    Weighted blend used for ordering and as the public ``similarity_score`` surrogate for L2 chunks.

    Base blend:
    - 0.42 provider relevance (search rank quality when present)
    - 0.33 source authority (domain tier + registry clamp)
    - 0.25 query/snippet lexical overlap (semantic proxy without extra embedding calls)

    Optional boosts/penalties (0–1 each): favor direct-answer and breed/topic-specific text;
    down-rank headings, image fragments, and boilerplate.
    """
    rel = normalize_provider_relevance(provider_relevance)
    auth = max(0.0, min(1.0, float(source_authority)))
    ov = max(0.0, min(1.0, float(query_overlap)))
    base = float(0.42 * rel + 0.33 * auth + 0.25 * ov)
    da = max(0.0, min(1.0, float(direct_answer_signal)))
    sp = max(0.0, min(1.0, float(specificity_signal)))
    nz = max(0.0, min(1.0, float(noise_signal)))
    adjusted = base + 0.09 * da + 0.06 * sp - 0.14 * nz
    return max(0.0, min(1.0, adjusted))