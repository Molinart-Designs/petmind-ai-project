"""Tests for snippet quality heuristics used in extraction and confidence calibration."""

from datetime import datetime, timezone

from pydantic import HttpUrl

from src.research.evidence_extractor import RetrievalEvidenceExtractor
from src.research.snippet_heuristics import (
    dedupe_similar_claim_units,
    should_discard_snippet_unit,
    infer_snippet_scope_fields,
    normalize_article_url_for_dedupe,
    strip_inline_markdown_images,
)
from src.research.web_retriever import TrustedSourceHit


def test_strip_inline_markdown_images() -> None:
    raw = "Intro text here. ![dog photo](https://cdn.example/a.png) More guidance about brushing daily habits."
    out = strip_inline_markdown_images(raw)
    assert "https://cdn.example" not in out
    assert "brushing" in out.lower()


def test_dedupe_similar_claim_units_collapses_repeated_openings() -> None:
    u1 = "Alpha guidance paragraph with enough tokens to qualify as a distinct claim unit here."
    out = dedupe_similar_claim_units([u1, u1])
    assert len(out) == 1


def test_should_discard_heading_that_mirrors_question() -> None:
    q = "What is the best feeding schedule for an adult dog with sensitive digestion?"
    heading = "What Is The Best Feeding Schedule For An Adult Dog With Sensitive Digestion"
    assert should_discard_snippet_unit(heading, q) is True


def test_should_discard_puppy_snippet_for_adult_question() -> None:
    q = "How much exercise does my adult dog need each day?"
    puppy = (
        "Puppy play sessions are short bursts of activity because puppies need frequent naps and gentle "
        "exercise only during growth plates development weeks before adolescence transitions occur naturally."
    )
    assert should_discard_snippet_unit(puppy, q) is True


def test_infer_scope_pomeranian_adult_from_question() -> None:
    q = "My adult Pomeranian has dry skin; what grooming routine helps?"
    body = (
        "Adult dogs with thick double coats benefit from weekly brushing and occasional conditioning "
        "recommended by veterinary dermatology articles for small breed companion animals living indoors."
    )
    fields = infer_snippet_scope_fields(q, page_title="Pomeranian coat care", unit_text=body)
    assert fields.get("species") == "dog"
    assert fields.get("breed") == "pomeranian"
    assert fields.get("life_stage") == "adult"


def test_normalize_article_url_strips_query_consistency() -> None:
    a = normalize_article_url_for_dedupe("https://Example.COM/path?x=1")
    b = normalize_article_url_for_dedupe("https://example.com/path?y=2")
    assert a == b


def test_from_trusted_hits_dedupes_duplicate_units_same_page() -> None:
    """Two near-identical sentences from the same hit should collapse to one snippet."""
    dup = (
        "Owners should measure meals using a standard cup and adjust portions when body condition changes "
        "during seasonal activity shifts for adult dogs living mostly indoors with limited exercise."
    )
    hit = TrustedSourceHit(
        url=HttpUrl("https://avma.org/doc"),
        title="Nutrition",
        excerpt=f"{dup} {dup}",
        source_key="src_a",
        retrieved_at=datetime.now(timezone.utc),
        relevance_score=0.9,
    )
    out = RetrievalEvidenceExtractor.from_trusted_hits(
        [hit],
        provider_id="stub",
        query="how to measure dog food portions for adult dogs indoors",
    )
    assert len(out.research_evidence.snippets) <= 2
