"""
Lightweight species / breed / life_stage hints for external evidence envelopes.

Conservative: skipped when ``avoid_breed_life_inference`` is True (medical-sensitive flows).
"""

from __future__ import annotations

import re


def _norm(q: str) -> str:
    return " ".join(q.lower().strip().split())


def infer_scope_from_question(
    question: str,
    *,
    avoid_breed_life_inference: bool = False,
) -> tuple[str | None, str | None, str | None]:
    """
    Return ``(species, breed, life_stage)`` tuples with high-precision heuristics only.

    When ``avoid_breed_life_inference`` is True (e.g. medical topic), only species may be filled
    when the question explicitly names dog/cat.
    """
    q = _norm(question)
    species: str | None = None
    breed: str | None = None
    life_stage: str | None = None

    if re.search(r"\b(perro|perros|dog|dogs|canine)\b", q) and not re.search(r"\b(gato|gatos|cat|cats|feline)\b", q):
        species = "dog"
    elif re.search(r"\b(gato|gatos|cat|cats|feline)\b", q) and not re.search(r"\b(perro|dog)\b", q):
        species = "cat"

    if avoid_breed_life_inference:
        return species, None, None

    if re.search(r"\bpomerania|pomeranian|pom\s+dog\b", q):
        breed = "pomeranian"

    if re.search(r"\b(puppy|puppies|cachorro|cachorros)\b", q):
        life_stage = "puppy"
    elif re.search(r"\b(senior|geriatric|anciano|anciana|viejo|vieja)\b", q):
        life_stage = "senior"
    elif re.search(r"\b(\d+)\s*(años?|years?)\s*old\b", q) or re.search(r"\b(adult|adulto|adulta)\b", q):
        life_stage = "adult"

    return species, breed, life_stage


def merge_research_scope(
    *,
    topic: str | None,
    species: str | None,
    breed: str | None,
    life_stage: str | None,
    question: str,
    is_medical_topic: bool,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Fill missing envelope fields from question heuristics without overwriting explicit profile data."""
    inf_s, inf_b, inf_ls = infer_scope_from_question(
        question,
        avoid_breed_life_inference=is_medical_topic,
    )
    out_species = species or inf_s
    out_breed = breed or inf_b
    out_life = life_stage or inf_ls
    return topic, out_species, out_breed, out_life
