"""Tests for external research candidate persistence helpers."""

from datetime import datetime, timezone

import pytest
from pydantic import HttpUrl

from src.research.evidence_extractor import (
    EvidenceExtractionResult,
    FrontendEvidenceBundle,
    ReviewDraftEvidenceBundle,
)
from src.research.ingest_candidates import (
    ExternalResearchIngestInput,
    NullExternalResearchPersistence,
    build_separated_evidence_json,
    classify_external_candidate_state,
    ingest_external_research_candidate,
)
from src.research.schemas import (
    ExternalSource,
    ExternalSourceType,
    ExtractedSnippet,
    KnowledgeRecordStatus,
    ResearchEvidence,
)


def _minimal_extraction() -> EvidenceExtractionResult:
    now = datetime.now(timezone.utc)
    src = ExternalSource(
        id="00000000-0000-4000-8000-000000000099",
        source_key="src1",
        base_url=HttpUrl("https://avma.org/"),
        authority_score=0.7,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )
    sn1 = ExtractedSnippet(
        id="snippet-1",
        external_source_id=src.id,
        text=(
            "Enough substantive text for a first snippet body here, including multiple words "
            "about routine preventive care and appetite monitoring for adult companion animals."
        ),
        authority_score=0.7,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )
    sn2 = ExtractedSnippet(
        id="snippet-2",
        external_source_id=src.id,
        text=(
            "A second excerpt continues with hydration guidance, exercise tolerance, and when to "
            "escalate concerns to a licensed veterinarian during home observation periods."
        ),
        authority_score=0.7,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )
    ev = ResearchEvidence(snippets=[sn1, sn2], sources=[src])
    fe = FrontendEvidenceBundle(citations=[], generated_at=now)
    rd = ReviewDraftEvidenceBundle(
        snippets=[sn1, sn2],
        claims=[],
        sources=[src],
        provider_id="stub",
        generated_at=now,
    )
    return EvidenceExtractionResult(frontend=fe, review_draft=rd, research_evidence=ev)


def test_classify_provisional_by_default() -> None:
    status, blocked = classify_external_candidate_state(
        "general",
        topic="daily walks",
        species="dog",
        life_stage="adult",
        category="exercise",
    )
    assert status == KnowledgeRecordStatus.provisional.value
    assert blocked is False


def test_classify_needs_review_for_medical_sensitivity() -> None:
    status, blocked = classify_external_candidate_state(
        "medical",
        topic=None,
        species=None,
        life_stage=None,
        category=None,
    )
    assert status == KnowledgeRecordStatus.needs_review.value
    assert blocked is True


def test_classify_needs_review_for_heuristic_topic() -> None:
    status, blocked = classify_external_candidate_state(
        "general",
        topic="Possible cancer screening for senior pet",
        species=None,
        life_stage=None,
        category=None,
    )
    assert status == KnowledgeRecordStatus.needs_review.value
    assert blocked is True


def test_evidence_json_separates_top_level_keys() -> None:
    ext = _minimal_extraction()
    inp = ExternalResearchIngestInput(
        extraction=ext,
        frontend_answer_text="Owner-safe short reply.",
        internal_review_llm_draft="AI-only staff notes.",
    )
    env = build_separated_evidence_json(inp)
    assert env["schema_version"] == 2
    ev = env["evidence"]
    assert len(ev["records"]) == 2
    rec = ev["records"][0]
    assert rec["snippet_id"]
    assert rec["snippet"]
    assert rec["source_url"].startswith("http")
    assert rec["source_domain"] == "avma.org"
    assert rec["authority_score"] == pytest.approx(0.7)
    assert len(ev["research_evidence"]["snippets"]) == 2
    assert ev["retrieval_extraction_bundle"]["provider_id"] == "stub"
    syn = env["synthesis"]
    assert syn["frontend_answer"]["safe_for_api"] is True
    assert syn["frontend_answer"]["text"] == "Owner-safe short reply."
    rd = syn["review_draft"]
    assert rd["text"] == "AI-only staff notes."
    assert rd["ai_generated"] is True
    assert rd["provisional"] is True
    assert rd["not_evidence"] is True


def test_evidence_json_legacy_synthesis_text_used_when_internal_draft_missing() -> None:
    ext = _minimal_extraction()
    inp = ExternalResearchIngestInput(extraction=ext, synthesis_text="  legacy review  ")
    env = build_separated_evidence_json(inp)
    assert env["synthesis"]["review_draft"]["text"] == "legacy review"


@pytest.mark.asyncio
async def test_ingest_external_with_null_store() -> None:
    inp = ExternalResearchIngestInput(extraction=_minimal_extraction())
    out = await ingest_external_research_candidate(inp, store=NullExternalResearchPersistence())
    assert out["research_candidate_id"] == 0
    assert out["document_chunks_ingested"] == 0
    assert out["status"] == "noop"


@pytest.mark.asyncio
async def test_ingest_skips_when_placeholder_domain_or_low_quality() -> None:
    now = datetime.now(timezone.utc)
    src = ExternalSource(
        id="00000000-0000-4000-8000-000000000088",
        source_key="src1",
        base_url=HttpUrl("https://example.com/"),
        authority_score=0.7,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )
    sn1 = ExtractedSnippet(
        id="snippet-bad-1",
        external_source_id=src.id,
        text="x",
        authority_score=0.7,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )
    sn2 = ExtractedSnippet(
        id="snippet-bad-2",
        external_source_id=src.id,
        text="y",
        authority_score=0.7,
        source_type=ExternalSourceType.allowlisted_web,
        retrieved_at=now,
    )
    ev = ResearchEvidence(snippets=[sn1, sn2], sources=[src])
    fe = FrontendEvidenceBundle(citations=[], generated_at=now)
    rd = ReviewDraftEvidenceBundle(
        snippets=[sn1, sn2],
        claims=[],
        sources=[src],
        provider_id="stub",
        generated_at=now,
    )
    bad = EvidenceExtractionResult(frontend=fe, review_draft=rd, research_evidence=ev)
    inp = ExternalResearchIngestInput(extraction=bad)
    out = await ingest_external_research_candidate(inp, store=NullExternalResearchPersistence())
    assert out["status"] == "skipped_evidence_quality_gate"
    assert out.get("gate_rejection_reasons")
