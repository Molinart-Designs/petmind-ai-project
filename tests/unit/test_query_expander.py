"""Unit tests for research query expansion."""

import pytest

from src.api.schemas import PetProfile, QueryFilters
from src.research.query_expander import (
    DeterministicResearchQueryExpander,
    HybridResearchQueryExpander,
)


def test_expand_returns_between_three_and_seven() -> None:
    exp = DeterministicResearchQueryExpander()
    out = exp.expand("How much exercise does my pet need each day?")
    assert 3 <= len(out) <= 7
    assert all(q.strip() for q in out)


def test_expand_includes_species_breed_life_stage_topics() -> None:
    exp = DeterministicResearchQueryExpander()
    profile = PetProfile(
        species="dog",
        breed="Border Collie",
        life_stage="adult",
    )
    filters = QueryFilters(category="exercise", species=None, life_stage=None)
    out = exp.expand(
        "How much daily activity is appropriate?",
        pet_profile=profile,
        filters=filters,
    )
    joined = " ".join(out).lower()
    assert "dog" in joined
    assert "border collie" in joined
    assert "adult" in joined
    assert "exercise" in joined


def test_conditions_not_injected_into_queries() -> None:
    exp = DeterministicResearchQueryExpander()
    profile = PetProfile(
        species="cat",
        conditions=["diabetes mellitus", "chronic kidney disease"],
    )
    out = exp.expand("What should I discuss with my veterinarian about diet?", pet_profile=profile)
    blob = " ".join(out).lower()
    assert "diabetes" not in blob
    assert "kidney" not in blob


def test_filters_species_when_profile_missing() -> None:
    exp = DeterministicResearchQueryExpander()
    filters = QueryFilters(species="rabbit", life_stage="senior", category=None)
    out = exp.expand("Housing and enrichment ideas", filters=filters)
    joined = " ".join(out).lower()
    assert "rabbit" in joined
    assert "senior" in joined


def test_invalid_min_max_raises() -> None:
    exp = DeterministicResearchQueryExpander()
    with pytest.raises(ValueError):
        exp.expand("question", min_queries=5, max_queries=3)


def test_hybrid_falls_back_when_llm_returns_none() -> None:
    class _NoLLM:
        def try_expand(self, **kwargs: object) -> None:
            return None

    hybrid = HybridResearchQueryExpander(llm_backend=_NoLLM())
    det = DeterministicResearchQueryExpander()
    q = "Basic grooming routine for a short-haired pet"
    assert hybrid.expand(q) == det.expand(q)


def test_hybrid_uses_llm_when_valid() -> None:
    class _FixedLLM:
        def try_expand(self, **kwargs: object) -> list[str]:
            return ["a", "b", "c", "d"]

    hybrid = HybridResearchQueryExpander(llm_backend=_FixedLLM())
    out = hybrid.expand("anything")
    assert out == ["a", "b", "c", "d"]


def test_hybrid_ignores_llm_when_too_short() -> None:
    class _ShortLLM:
        def try_expand(self, **kwargs: object) -> list[str]:
            return ["only", "two"]

    hybrid = HybridResearchQueryExpander(llm_backend=_ShortLLM())
    out = hybrid.expand("Need at least three distinct retrieval strings for this topic please")
    assert len(out) >= 3


def test_noop_delegates_to_deterministic() -> None:
    from src.research.query_expander import NoOpQueryExpander

    noop = NoOpQueryExpander()
    det = DeterministicResearchQueryExpander()
    q = "Water intake during hot weather"
    assert noop.expand(q) == det.expand(q)
