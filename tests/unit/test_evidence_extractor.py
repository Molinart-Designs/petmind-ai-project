"""Tests for retrieval → structured evidence extraction."""

from datetime import datetime, timezone

from pydantic import HttpUrl

from src.research.evidence_extractor import (
    NullEvidenceExtractor,
    PageEvidenceExtractor,
    RetrievalEvidenceExtractor,
    normalize_whitespace,
    split_excerpt_into_claim_units,
)
from src.research.schemas import ExternalSource, ExternalSourceType
from src.research.web_retriever import BlockedUrl, FetchedPage, TrustedRetrievalResult, TrustedSourceHit


def test_normalize_whitespace() -> None:
    assert normalize_whitespace("  a   \n\n  b  ") == "a b"


def test_split_excerpt_into_multiple_units() -> None:
    text = (
        "First sentence here with enough characters to pass the minimum unit length threshold. "
        "Second one follows with similar length so the splitter keeps it as its own claim unit! "
        "Third is last and must also exceed twenty eight characters for inclusion in the output?"
    )
    units = split_excerpt_into_claim_units(text, max_units=8)
    assert len(units) == 3
    assert units[0].startswith("First")


def test_from_trusted_hits_builds_frontend_and_review() -> None:
    hit = TrustedSourceHit(
        url=HttpUrl("https://avma.org/doc"),
        title="Example doc",
        excerpt=(
            "Alpha statement expands with enough veterinary context words to satisfy snippet quality. "
            "Beta statement continues with hydration guidance and exercise tolerance for adult dogs."
        ),
        source_key="src_a",
        retrieved_at=datetime.now(timezone.utc),
    )
    out = RetrievalEvidenceExtractor.from_trusted_hits([hit], provider_id="stub")
    assert len(out.frontend.citations) == 2
    assert len(out.review_draft.claims) == 2
    assert len(out.research_evidence.snippets) == 2
    assert out.frontend.is_provisional is True
    assert "answer" not in out.frontend.model_dump_json().lower()
    assert out.review_draft.blocked_url_count == 0
    c0 = out.frontend.citations[0]
    assert c0.source_url.startswith("https://")
    assert c0.source_key == "src_a"
    assert c0.snippet_id == out.research_evidence.snippets[0].id


def test_from_retrieval_result_counts_blocked() -> None:
    hit = TrustedSourceHit(
        url=HttpUrl("https://example.com/"),
        title=None,
        excerpt="Single paragraph without extra punctuation marks for splitting purposes here",
        source_key="k",
        retrieved_at=datetime.now(timezone.utc),
    )
    res = TrustedRetrievalResult(
        hits=[hit],
        blocked=[BlockedUrl(url="https://evil.test/", reason="domain_not_allowlisted")],
        provider_id="stub",
    )
    out = RetrievalEvidenceExtractor.from_retrieval_result(res)
    assert out.review_draft.blocked_url_count == 1
    assert len(out.research_evidence.snippets) >= 1


def test_sources_by_key_used_for_external_source_id() -> None:
    src = ExternalSource(
        id="11111111-1111-4111-8111-111111111111",
        source_key="reg",
        base_url=HttpUrl("https://avma.org/"),
        authority_score=0.9,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=datetime.now(timezone.utc),
    )
    hit = TrustedSourceHit(
        url=HttpUrl("https://avma.org/p"),
        excerpt=(
            "One claim only would be too short; this paragraph adds measured portions, body condition "
            "scoring, and veterinary follow up so the extraction pipeline keeps a single solid unit."
        ),
        source_key="reg",
        retrieved_at=datetime.now(timezone.utc),
    )
    out = RetrievalEvidenceExtractor.from_trusted_hits(
        [hit],
        provider_id="p",
        sources_by_key={"reg": src},
    )
    assert out.research_evidence.snippets[0].external_source_id == src.id


def test_page_evidence_extractor_respects_max_snippets() -> None:
    src = ExternalSource(
        id="22222222-2222-4222-8222-222222222222",
        source_key="s",
        base_url=HttpUrl("https://example.com/"),
        authority_score=0.5,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=datetime.now(timezone.utc),
    )
    page = FetchedPage(
        url=HttpUrl("https://example.com/x"),
        source_key="s",
        body_text=(
            "First sentence is definitely long enough for the splitter to keep it. "
            "Second sentence also has enough characters to form another unit."
        ),
        retrieved_at=datetime.now(timezone.utc),
    )
    ext = PageEvidenceExtractor()
    snippets = ext.extract(page=page, source=src, query_hints=[], max_snippets=2)
    assert len(snippets) == 2


def test_null_evidence_extractor_empty() -> None:
    src = ExternalSource(
        id="33333333-3333-4333-8333-333333333333",
        source_key="s",
        base_url=HttpUrl("https://example.com/"),
        authority_score=0.5,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=datetime.now(timezone.utc),
    )
    page = FetchedPage(
        url=HttpUrl("https://example.com/"),
        source_key="s",
        body_text="x",
        retrieved_at=datetime.now(timezone.utc),
    )
    assert NullEvidenceExtractor().extract(page=page, source=src, query_hints=[], max_snippets=5) == []


def test_empty_extraction_bundle() -> None:
    empty = NullEvidenceExtractor.empty_extraction()
    assert empty.research_evidence.snippets == []
    assert empty.frontend.citations == []
