"""Unit tests for snippet quality heuristics and persistence eligibility gates."""

from datetime import datetime, timezone

import pytest
from pydantic import HttpUrl

from src.research.evidence_extractor import RetrievalEvidenceExtractor
from src.research.evidence_quality import (
    evidence_bundle_eligible_for_persistence,
    is_placeholder_evidence_hostname,
    snippet_text_meets_evidence_quality,
)
from src.research.schemas import (
    ExternalSource,
    ExternalSourceType,
    ExtractedSnippet,
    ResearchEvidence,
)


def test_placeholder_host_detection() -> None:
    assert is_placeholder_evidence_hostname("example.com") is True
    assert is_placeholder_evidence_hostname("www.EXAMPLE.COM") is True
    assert is_placeholder_evidence_hostname("avma.org") is False


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", False),
        ("short.", False),
        ("word " * 3, False),
        (
            "This sentence has enough tokens and length to pass the minimum bar for external evidence.",
            True,
        ),
    ],
)
def test_snippet_text_quality_heuristic(text: str, expected: bool) -> None:
    assert snippet_text_meets_evidence_quality(text) is expected


def test_evidence_bundle_rejects_placeholder_domain_even_with_long_snippets() -> None:
    now = datetime.now(timezone.utc)
    src = ExternalSource(
        id="00000000-0000-4000-8000-000000000001",
        source_key="k",
        base_url=HttpUrl("https://example.com/"),
        authority_score=0.8,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )
    body = (
        "First long excerpt about canine hydration and appetite monitoring during mild illness "
        "with enough vocabulary diversity to satisfy snippet quality heuristics in the pipeline."
    )
    body2 = (
        "Second long excerpt about gradual return to exercise and when to contact a veterinarian "
        "if vomiting persists beyond twenty four hours or the animal appears painful at rest."
    )
    sn1 = ExtractedSnippet(
        id="s1",
        external_source_id=src.id,
        text=body,
        authority_score=0.8,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )
    sn2 = ExtractedSnippet(
        id="s2",
        external_source_id=src.id,
        text=body2,
        authority_score=0.8,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )
    ev = ResearchEvidence(snippets=[sn1, sn2], sources=[src])
    ex = RetrievalEvidenceExtractor.from_research_evidence(ev, provider_id="stub", blocked_url_count=0)
    ok, reasons = evidence_bundle_eligible_for_persistence(ex)
    assert ok is False
    assert any("placeholder_domain" in r for r in reasons)


def test_evidence_bundle_rejects_reddit_only_sources() -> None:
    now = datetime.now(timezone.utc)
    src = ExternalSource(
        id="00000000-0000-4000-8000-000000000001",
        source_key="k",
        base_url=HttpUrl("https://reddit.com/"),
        authority_score=0.24,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )
    body = (
        "Forum discussion paragraph with enough tokens to pass snippet quality heuristics and "
        "describe anecdotal walking habits that are not structured veterinary references at all."
    )
    body2 = (
        "Second paragraph continues anecdotal theme with hydration bowls summer heat and timing "
        "of walks so that quality checks pass while domains remain purely social discussion boards."
    )
    sn1 = ExtractedSnippet(
        id="s1",
        external_source_id=src.id,
        text=body,
        authority_score=0.24,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
        evidence_page_url="https://www.reddit.com/r/dogs/a",
    )
    sn2 = ExtractedSnippet(
        id="s2",
        external_source_id=src.id,
        text=body2,
        authority_score=0.24,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
        evidence_page_url="https://old.reddit.com/r/dogs/b",
    )
    ev = ResearchEvidence(snippets=[sn1, sn2], sources=[src])
    ex = RetrievalEvidenceExtractor.from_research_evidence(ev, provider_id="stub", blocked_url_count=0)
    ok, reasons = evidence_bundle_eligible_for_persistence(ex)
    assert ok is False
    assert any("reddit_requires_anchor_trusted_snippet" in r for r in reasons)


def test_evidence_bundle_rejects_reddit_when_companion_not_anchor_tier() -> None:
    """Reddit + a second host below anchor authority does not satisfy the mix rule."""
    now = datetime.now(timezone.utc)
    reddit_src = ExternalSource(
        id="00000000-0000-4000-8000-000000000001",
        source_key="r",
        base_url=HttpUrl("https://reddit.com/"),
        authority_score=0.24,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )
    low_src = ExternalSource(
        id="00000000-0000-4000-8000-000000000002",
        source_key="l",
        base_url=HttpUrl("https://low-tier-blog.pet/"),
        authority_score=0.2,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )
    body_r = (
        "Forum discussion paragraph with enough tokens to pass snippet quality heuristics and "
        "describe anecdotal walking habits that are not structured veterinary references at all."
    )
    body_l = (
        "Low authority blog post with enough tokens about dog toys and chewing habits during "
        "teething phases so quality checks pass while authority stays below anchor threshold here."
    )
    sn_r = ExtractedSnippet(
        id="s-r",
        external_source_id=reddit_src.id,
        text=body_r,
        authority_score=0.24,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
        evidence_page_url="https://www.reddit.com/r/dogs/a",
    )
    sn_l = ExtractedSnippet(
        id="s-l",
        external_source_id=low_src.id,
        text=body_l,
        authority_score=0.30,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
        evidence_page_url="https://low-tier-blog.pet/p/1",
    )
    ev = ResearchEvidence(snippets=[sn_r, sn_l], sources=[reddit_src, low_src])
    ex = RetrievalEvidenceExtractor.from_research_evidence(ev, provider_id="stub", blocked_url_count=0)
    ok, reasons = evidence_bundle_eligible_for_persistence(ex)
    assert ok is False
    assert any("reddit_requires_anchor_trusted_snippet" in r for r in reasons)


def test_evidence_bundle_accepts_reddit_with_anchor_trusted_snippet() -> None:
    now = datetime.now(timezone.utc)
    reddit_src = ExternalSource(
        id="00000000-0000-4000-8000-000000000001",
        source_key="r",
        base_url=HttpUrl("https://reddit.com/"),
        authority_score=0.24,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )
    avma_src = ExternalSource(
        id="00000000-0000-4000-8000-000000000002",
        source_key="a",
        base_url=HttpUrl("https://avma.org/"),
        authority_score=0.82,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )
    body_r = (
        "Forum discussion paragraph with enough tokens to pass snippet quality heuristics and "
        "describe anecdotal walking habits that are not structured veterinary references at all."
    )
    body_a = (
        "American Veterinary Medical Association aligned guidance stresses wellness visits and "
        "parasite prevention tailored to regional risks for adult dogs living mostly indoors."
    )
    sn_r = ExtractedSnippet(
        id="s-r",
        external_source_id=reddit_src.id,
        text=body_r,
        authority_score=0.24,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
        evidence_page_url="https://www.reddit.com/r/dogs/a",
    )
    sn_a = ExtractedSnippet(
        id="s-a",
        external_source_id=avma_src.id,
        text=body_a,
        authority_score=0.82,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
        evidence_page_url="https://www.avma.org/resources/pet-care/wellness",
    )
    ev = ResearchEvidence(snippets=[sn_r, sn_a], sources=[reddit_src, avma_src])
    ex = RetrievalEvidenceExtractor.from_research_evidence(ev, provider_id="stub", blocked_url_count=0)
    ok, reasons = evidence_bundle_eligible_for_persistence(ex)
    assert ok is True
    assert reasons == []


def test_evidence_bundle_accepts_avma_org_pair() -> None:
    now = datetime.now(timezone.utc)
    src = ExternalSource(
        id="00000000-0000-4000-8000-000000000001",
        source_key="k",
        base_url=HttpUrl("https://avma.org/"),
        authority_score=0.8,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )
    body = (
        "American Veterinary Medical Association aligned guidance stresses wellness visits and "
        "parasite prevention tailored to regional risks for adult dogs living mostly indoors."
    )
    body2 = (
        "Nutrition paragraphs should still mention measuring cups, body condition scoring, and "
        "treat allocation so owners avoid accidental caloric surplus during training sessions."
    )
    sn1 = ExtractedSnippet(
        id="s1",
        external_source_id=src.id,
        text=body,
        authority_score=0.8,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )
    sn2 = ExtractedSnippet(
        id="s2",
        external_source_id=src.id,
        text=body2,
        authority_score=0.8,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )
    ev = ResearchEvidence(snippets=[sn1, sn2], sources=[src])
    ex = RetrievalEvidenceExtractor.from_research_evidence(ev, provider_id="stub", blocked_url_count=0)
    ok, reasons = evidence_bundle_eligible_for_persistence(ex)
    assert ok is True
    assert reasons == []
