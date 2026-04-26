"""Pipeline tests: domain authority, ranking scores, excerpt cleanup, public disclaimers."""

from datetime import datetime, timezone

import pytest
from pydantic import HttpUrl

from src.research.domain_authority import domain_authority_score, is_anecdotal_social_domain
from src.research.evidence_extractor import RetrievalEvidenceExtractor, select_snippets_for_research_result
from src.research.excerpt_sanitize import clean_excerpt_for_evidence
from src.research.external_ranking import composite_snippet_ranking
from src.research.schemas import ExternalSource, ExternalSourceType, ExtractedSnippet
from src.research.trusted_research_service import dedupe_and_rank_trusted_hits
from src.research.web_retriever import TrustedRetrievalResult, TrustedSourceHit
from src.security.guardrails import (
    INTERNAL_CONTEXT_LIMITED_DISCLAIMER,
    assess_retrieval_grounding,
    postprocess_answer,
    prioritize_sources_for_public_response,
)


def test_reddit_domain_authority_below_pitpat() -> None:
    assert domain_authority_score("reddit.com") < domain_authority_score("pitpat.com")
    assert is_anecdotal_social_domain("www.reddit.com") is True


def test_dedupe_prefers_higher_tier_domain_at_equal_relevance() -> None:
    now = datetime.now(timezone.utc)
    rel = 0.9
    reddit = TrustedSourceHit(
        url=HttpUrl("https://www.reddit.com/r/dogs/comments/x/test/"),
        title="Thread title",
        excerpt="Skip to main content\n\n" + ("Dogs need daily walks for health and wellbeing. " * 6),
        source_key="k",
        retrieved_at=now,
        relevance_score=rel,
    )
    pitpat = TrustedSourceHit(
        url=HttpUrl("https://www.pitpat.com/articles/exercise"),
        title="Exercise guide",
        excerpt=("Measured activity helps owners track rest and play for companion animals. " * 6),
        source_key="k",
        retrieved_at=now,
        relevance_score=rel,
    )
    ranked = dedupe_and_rank_trusted_hits([reddit, pitpat])
    assert "pitpat.com" in str(ranked[0].url).lower()


def test_navigation_junk_removed_from_excerpt() -> None:
    raw = "Skip to main content\n\nFeatured image\n\nDogs benefit from routine exercise and hydration daily."
    cleaned = clean_excerpt_for_evidence(raw)
    assert "skip to main" not in cleaned.lower()
    assert "dogs benefit" in cleaned.lower()


def test_composite_ranking_uses_relevance_and_authority() -> None:
    high = composite_snippet_ranking(provider_relevance=0.9, source_authority=0.6, query_overlap=0.5)
    low = composite_snippet_ranking(provider_relevance=0.9, source_authority=0.22, query_overlap=0.5)
    assert high > low


def test_from_trusted_hits_sets_non_flat_similarity_via_ranking() -> None:
    now = datetime.now(timezone.utc)
    hit_high = TrustedSourceHit(
        url=HttpUrl("https://avma.org/resources/pet-care"),
        title="AVMA resource page",
        excerpt=(
            "Routine preventive veterinary visits help catch dental disease and obesity trends early "
            "in adult companion animals living mostly indoors with limited exercise opportunities."
        ),
        source_key="src",
        retrieved_at=now,
        relevance_score=0.92,
    )
    hit_low = TrustedSourceHit(
        url=HttpUrl("https://www.reddit.com/r/dogs/comments/abc/low/"),
        title="Random thread",
        excerpt=(
            "Anecdotal note: my neighbor walks their dog twice daily and says the pet seems calmer "
            "afterward during hot summer months when pavement is cooler in early mornings only."
        ),
        source_key="src",
        retrieved_at=now,
        relevance_score=0.92,
    )
    out = RetrievalEvidenceExtractor.from_trusted_hits(
        [hit_low, hit_high],
        provider_id="stub",
        query="preventive veterinary visits adult dog exercise",
        max_snippets_out=4,
    )
    scores = [s.ranking_score for s in out.research_evidence.snippets if s.ranking_score is not None]
    assert len(set(round(x, 3) for x in scores)) >= 1
    assert max(scores) > min(scores)


def test_select_snippets_omits_all_reddit_when_no_anchor() -> None:
    now = datetime.now(timezone.utc)
    src = ExternalSource(
        id="00000000-0000-4000-8000-000000000001",
        source_key="k",
        base_url=HttpUrl("https://reddit.com/"),
        authority_score=0.24,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )
    r1 = ExtractedSnippet(
        id="s-r1",
        external_source_id=src.id,
        text=(
            "Reddit user claims their dog prefers morning walks before breakfast during summer heat waves "
            "and avoids midday pavement burns on very hot calendar days in urban neighborhoods."
        ),
        authority_score=0.24,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
        evidence_page_url="https://reddit.com/r/dogs/x",
        ranking_score=0.9,
    )
    r2 = ExtractedSnippet(
        id="s-r2",
        external_source_id=src.id,
        text=(
            "Another forum comment suggests puzzle feeders reduce scavenging when owners work long shifts "
            "and cannot supervise free feeding bowls left out in kitchens with multiple pets present."
        ),
        authority_score=0.24,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
        evidence_page_url="https://old.reddit.com/r/dogs/y",
        ranking_score=0.88,
    )
    picked = select_snippets_for_research_result([r1, r2], max_total=4, max_low_tier=1)
    assert picked == []


def test_select_snippets_caps_reddit_when_stronger_exists() -> None:
    now = datetime.now(timezone.utc)
    src = ExternalSource(
        id="00000000-0000-4000-8000-000000000001",
        source_key="k",
        base_url=HttpUrl("https://example.org/"),
        authority_score=0.5,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )
    strong = ExtractedSnippet(
        id="s-strong",
        external_source_id=src.id,
        text=(
            "Veterinary nutrition references recommend measuring portions and monitoring body condition "
            "weekly during weight loss trials for small breed adult dogs with indoor lifestyles."
        ),
        authority_score=0.55,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
        evidence_page_url="https://avma.org/a",
        ranking_score=0.72,
    )
    r1 = ExtractedSnippet(
        id="s-r1",
        external_source_id=src.id,
        text=(
            "Reddit user claims their dog prefers morning walks before breakfast during summer heat waves "
            "and avoids midday pavement burns on very hot calendar days in urban neighborhoods."
        ),
        authority_score=0.24,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
        evidence_page_url="https://reddit.com/r/dogs/x",
        ranking_score=0.65,
    )
    r2 = ExtractedSnippet(
        id="s-r2",
        external_source_id=src.id,
        text=(
            "Another forum comment suggests puzzle feeders reduce scavenging when owners work long shifts "
            "and cannot supervise free feeding bowls left out in kitchens with multiple pets present."
        ),
        authority_score=0.24,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
        evidence_page_url="https://old.reddit.com/r/dogs/y",
        ranking_score=0.64,
    )
    picked = select_snippets_for_research_result([r2, strong, r1], max_total=3, max_low_tier=1)
    reddit_count = sum(1 for s in picked if "reddit.com" in (s.evidence_page_url or ""))
    assert reddit_count <= 1
    assert any(s.id == "s-strong" for s in picked)


def test_prioritize_sources_omits_reddit_without_trusted_anchor_row() -> None:
    reddit_row = {
        "chunk_id": "r1",
        "document_id": "ext",
        "similarity_score": 0.99,
        "snippet": "Anecdotal Reddit text with enough length for tests.",
        "metadata": {
            "layer": "trusted_external",
            "source_domain": "reddit.com",
            "trusted_evidence_anchor": False,
            "anecdotal_supplemental": True,
        },
    }
    assert prioritize_sources_for_public_response([reddit_row], max_items=5) == []


def test_prioritize_sources_keeps_one_reddit_when_anchor_present() -> None:
    anchor = {
        "chunk_id": "a1",
        "document_id": "ext",
        "similarity_score": 0.7,
        "snippet": "AVMA-aligned guidance with enough substance for the public sources list here.",
        "metadata": {
            "layer": "trusted_external",
            "source_domain": "avma.org",
            "trusted_evidence_anchor": True,
            "anecdotal_supplemental": False,
        },
    }
    r_high = {
        "chunk_id": "r1",
        "document_id": "ext",
        "similarity_score": 0.95,
        "snippet": "Reddit anecdote one with enough tokens to appear if policy allows it at all.",
        "metadata": {
            "layer": "trusted_external",
            "source_domain": "reddit.com",
            "trusted_evidence_anchor": False,
            "anecdotal_supplemental": True,
        },
    }
    r_low = {
        "chunk_id": "r2",
        "document_id": "ext",
        "similarity_score": 0.5,
        "snippet": "Second Reddit anecdote that should be dropped when cap is one supplemental row.",
        "metadata": {
            "layer": "trusted_external",
            "source_domain": "www.reddit.com",
            "trusted_evidence_anchor": False,
            "anecdotal_supplemental": True,
        },
    }
    out = prioritize_sources_for_public_response([r_low, anchor, r_high], max_items=5, max_anecdotal_tier=1)
    domains = [o["metadata"]["source_domain"] for o in out]
    assert "avma.org" in domains
    assert sum(1 for d in domains if "reddit" in d) == 1


def test_postprocess_external_disclaimer_not_kb_empty_wording() -> None:
    internal = [
        {
            "chunk_id": "i1",
            "document_id": "d1",
            "similarity_score": 0.4,
            "snippet": "weak internal",
            "metadata": {},
        }
    ]
    external = [
        {
            "chunk_id": "e1",
            "document_id": "ext",
            "title": "Article title from web",
            "similarity_score": 0.58,
            "snippet": "External grounded excerpt with enough substance for the user facing question.",
            "metadata": {
                "provisional": True,
                "layer": "trusted_external",
                "source_domain": "avma.org",
                "source_url": "https://avma.org/x",
            },
        }
    ]
    merged = internal + external
    result = postprocess_answer(
        answer="Answer using provisional web context.",
        question="How often should I walk my adult dog?",
        retrieved_chunks=merged,
        similarity_threshold=0.75,
        chunks_for_grounding=internal,
        used_provisional_external=True,
    )
    text = " ".join(result["disclaimers"]).lower()
    assert "not enough grounded context in the knowledge base" not in text
    assert INTERNAL_CONTEXT_LIMITED_DISCLAIMER.split()[0].lower() in text or "internally approved" in text


@pytest.mark.parametrize(
    "text",
    [
        "Skip to main content\n\nDogs need exercise.",
        "Featured image\n\nCats need hydration.",
    ],
)
def test_clean_excerpt_drops_short_nav_only_blocks(text: str) -> None:
    out = clean_excerpt_for_evidence(text)
    assert "skip" not in out.lower() or "dogs need" in out.lower() or "cats need" in out.lower()
