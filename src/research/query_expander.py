"""
Query expansion for trusted external retrieval.

Builds 3–7 focused search strings from the user question plus optional profile and filters.
Deterministic rules run first; an optional LLM backend can override when it returns a valid set.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.api.schemas import PetProfile, QueryFilters
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MIN_QUERIES = 3
_DEFAULT_MAX_QUERIES = 7
_MAX_QUERY_CHARS = 500

# Non-diagnostic care stems: general retrieval wording only (deterministic padding).
_SAFE_CARE_STEMS: tuple[str, ...] = (
    "daily care",
    "general information",
    "husbandry",
    "nutrition basics",
    "behavior basics",
)


def _normalize_question(question: str) -> str:
    text = " ".join(question.strip().split())
    return text


def _trim(s: str, max_len: int) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def _dedupe_preserve_order(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        key = " ".join(q.lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(_trim(q, _MAX_QUERY_CHARS))
    return out


def _species(profile: PetProfile | None, filters: QueryFilters | None) -> str | None:
    if profile and profile.species and profile.species.strip():
        return profile.species.strip()
    if filters and filters.species and filters.species.strip():
        return filters.species.strip()
    return None


def _life_stage(profile: PetProfile | None, filters: QueryFilters | None) -> str | None:
    if profile and profile.life_stage and profile.life_stage.strip():
        return profile.life_stage.strip()
    if filters and filters.life_stage and filters.life_stage.strip():
        return filters.life_stage.strip()
    return None


def _breed(profile: PetProfile | None) -> str | None:
    if profile and profile.breed and profile.breed.strip():
        return profile.breed.strip()
    return None


def _category(filters: QueryFilters | None) -> str | None:
    if filters and filters.category and filters.category.strip():
        return filters.category.strip()
    return None


def _deterministic_candidates(
    question: str,
    *,
    pet_profile: PetProfile | None,
    filters: QueryFilters | None,
) -> list[str]:
    """Build topic-oriented strings using only the question and structured fields (no conditions)."""
    full = _normalize_question(question)
    if not full:
        return []

    species = _species(pet_profile, filters)
    life_stage = _life_stage(pet_profile, filters)
    breed = _breed(pet_profile)
    category = _category(filters)

    candidates: list[str] = [full]

    if species:
        candidates.append(f"{species} {full}")
    if life_stage:
        candidates.append(f"{life_stage} pet {full}")
    if breed:
        candidates.append(f"{breed} {full}")
    if category:
        candidates.append(f"{category} {full}")
    if species and life_stage:
        candidates.append(f"{species} {life_stage} {full}")
    if species and breed:
        candidates.append(f"{species} {breed} {full}")

    # Truncation variants (same semantics, different length for retrieval diversity).
    if len(full) > 120:
        candidates.append(_trim(full, 120))
    if len(full) > 80:
        candidates.append(_trim(full, 80))
    if len(full) > 50:
        candidates.append(_trim(full, 50))

    return _dedupe_preserve_order(candidates)


def _pad_to_minimum(
    queries: list[str],
    question: str,
    *,
    min_queries: int,
    max_queries: int,
) -> list[str]:
    """Ensure at least min_queries using only the question text and safe generic stems."""
    base = _normalize_question(question)
    out = _dedupe_preserve_order(list(queries))

    if len(out) >= max_queries:
        return out[:max_queries]

    stem_idx = 0
    while len(out) < min_queries and stem_idx < len(_SAFE_CARE_STEMS):
        candidate = _trim(f"{base} {_SAFE_CARE_STEMS[stem_idx]}".strip(), _MAX_QUERY_CHARS)
        stem_idx += 1
        if candidate.lower() in {q.lower() for q in out}:
            continue
        out.append(candidate)

    n = 0
    while len(out) < min_queries and base:
        n += 1
        cut = max(8, len(base) - n * 5)
        piece = base[:cut].rstrip()
        if len(piece) < 5:
            break
        candidate = _trim(piece, _MAX_QUERY_CHARS)
        if candidate.lower() not in {q.lower() for q in out}:
            out.append(candidate)

    suffix = 0
    while len(out) < min_queries:
        suffix += 1
        candidate = _trim(f"{base} reference context {suffix}", _MAX_QUERY_CHARS)
        if candidate.lower() in {q.lower() for q in out}:
            if suffix > 50:
                break
            continue
        out.append(candidate)

    return out[:max_queries]


@runtime_checkable
class QueryExpander(Protocol):
    """Turn a user question into bounded research sub-queries."""

    def expand(
        self,
        question: str,
        *,
        pet_profile: PetProfile | None = None,
        filters: QueryFilters | None = None,
        min_queries: int = _DEFAULT_MIN_QUERIES,
        max_queries: int = _DEFAULT_MAX_QUERIES,
    ) -> list[str]:
        """
        Return ``min_queries``–``max_queries`` normalized search strings.

        Implementations must not fabricate clinical findings; profile ``conditions`` are ignored here.
        """


@runtime_checkable
class LLMResearchQueryExpansionBackend(Protocol):
    """Optional LLM-backed expansion; return None to keep deterministic results."""

    def try_expand(
        self,
        *,
        question: str,
        pet_profile: PetProfile | None,
        filters: QueryFilters | None,
        min_queries: int,
        max_queries: int,
    ) -> list[str] | None:
        """Return between min and max queries inclusive, or None on skip / failure."""


def _validate_llm_batch(
    batch: list[str] | None,
    *,
    min_queries: int,
    max_queries: int,
) -> list[str] | None:
    if not batch:
        return None
    cleaned = [_trim(q.strip(), _MAX_QUERY_CHARS) for q in batch if q and q.strip()]
    if len(cleaned) < min_queries:
        return None
    cleaned = _dedupe_preserve_order(cleaned)
    if len(cleaned) < min_queries:
        return None
    return cleaned[:max_queries]


class DeterministicResearchQueryExpander:
    """
    Rule-based expansion: species, life stage, breed, category, plus truncation variants.

    Does not use ``PetProfile.conditions`` or free-text ``notes`` to avoid steering toward
    unstated diagnoses.
    """

    def expand(
        self,
        question: str,
        *,
        pet_profile: PetProfile | None = None,
        filters: QueryFilters | None = None,
        min_queries: int = _DEFAULT_MIN_QUERIES,
        max_queries: int = _DEFAULT_MAX_QUERIES,
    ) -> list[str]:
        if min_queries < 1 or max_queries < min_queries:
            raise ValueError("min_queries and max_queries must satisfy 1 <= min_queries <= max_queries.")
        if max_queries > 20:
            raise ValueError("max_queries is capped for safety.")

        core = _normalize_question(question)
        if not core:
            return []

        candidates = _deterministic_candidates(core, pet_profile=pet_profile, filters=filters)
        out = _dedupe_preserve_order(candidates)
        if len(out) < min_queries:
            out = _pad_to_minimum(out, core, min_queries=min_queries, max_queries=max_queries)
        return out[:max_queries]


class HybridResearchQueryExpander:
    """
    Runs an optional ``LLMResearchQueryExpansionBackend`` first; falls back to deterministic.

    LLM output is accepted only if it yields at least ``min_queries`` non-empty strings.
    """

    def __init__(
        self,
        *,
        deterministic: DeterministicResearchQueryExpander | None = None,
        llm_backend: LLMResearchQueryExpansionBackend | None = None,
    ) -> None:
        self._deterministic = deterministic or DeterministicResearchQueryExpander()
        self._llm_backend = llm_backend

    def expand(
        self,
        question: str,
        *,
        pet_profile: PetProfile | None = None,
        filters: QueryFilters | None = None,
        min_queries: int = _DEFAULT_MIN_QUERIES,
        max_queries: int = _DEFAULT_MAX_QUERIES,
    ) -> list[str]:
        if self._llm_backend is not None:
            try:
                llm_raw = self._llm_backend.try_expand(
                    question=question,
                    pet_profile=pet_profile,
                    filters=filters,
                    min_queries=min_queries,
                    max_queries=max_queries,
                )
            except Exception:
                logger.warning("LLM query expansion failed; using deterministic expander.", exc_info=True)
                llm_raw = None
            validated = _validate_llm_batch(llm_raw, min_queries=min_queries, max_queries=max_queries)
            if validated is not None:
                return validated

        return self._deterministic.expand(
            question,
            pet_profile=pet_profile,
            filters=filters,
            min_queries=min_queries,
            max_queries=max_queries,
        )


class NoOpQueryExpander:
    """
    Backwards-compatible name: delegates to :class:`DeterministicResearchQueryExpander`.

    There is no separate "single-query" mode; external research expects several strings.
    """

    def __init__(self) -> None:
        self._inner = DeterministicResearchQueryExpander()

    def expand(
        self,
        question: str,
        *,
        pet_profile: PetProfile | None = None,
        filters: QueryFilters | None = None,
        min_queries: int = _DEFAULT_MIN_QUERIES,
        max_queries: int = _DEFAULT_MAX_QUERIES,
    ) -> list[str]:
        return self._inner.expand(
            question,
            pet_profile=pet_profile,
            filters=filters,
            min_queries=min_queries,
            max_queries=max_queries,
        )
