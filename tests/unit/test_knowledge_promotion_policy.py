"""Tests for explicit knowledge promotion policy and lifecycle transitions."""

import pytest

from src.research.knowledge_promotion_policy import (
    KnowledgePromotionError,
    assert_initial_external_status_never_approved,
    assert_refresh_status_never_auto_approved,
    assert_sensitive_medical_bundle_policies,
    initial_status_for_external_fallback,
    is_medical_or_heuristic_gated_lane,
    transition_needs_review_to_approved,
    transition_provisional_to_approved,
    transition_provisional_to_expired,
    transition_provisional_to_needs_review,
)
from src.research.schemas import KnowledgeRecordStatus


def test_initial_external_fallback_never_approved_general() -> None:
    status, blocked = initial_status_for_external_fallback(
        "general",
        topic="walks",
        species="dog",
        life_stage="adult",
        category="exercise",
    )
    assert status == KnowledgeRecordStatus.provisional.value
    assert status != KnowledgeRecordStatus.approved.value
    assert blocked is False
    assert_initial_external_status_never_approved(status)


def test_initial_external_fallback_medical_lane() -> None:
    status, blocked = initial_status_for_external_fallback(
        "medical",
        topic=None,
        species=None,
        life_stage=None,
        category=None,
    )
    assert status == KnowledgeRecordStatus.needs_review.value
    assert blocked is True
    assert_initial_external_status_never_approved(status)


def test_initial_external_fallback_heuristic_cancer_topic() -> None:
    status, blocked = initial_status_for_external_fallback(
        "general",
        topic="Cancer follow-up care",
        species=None,
        life_stage=None,
        category=None,
    )
    assert status == KnowledgeRecordStatus.needs_review.value
    assert blocked is True


def test_assert_initial_rejects_approved() -> None:
    with pytest.raises(KnowledgePromotionError, match="must not be persisted as approved"):
        assert_initial_external_status_never_approved(KnowledgeRecordStatus.approved.value)


def test_assert_refresh_status_never_auto_approved_rejects_approved() -> None:
    with pytest.raises(KnowledgePromotionError, match="refresh workers"):
        assert_refresh_status_never_auto_approved(KnowledgeRecordStatus.approved.value)


def test_transition_provisional_to_approved_manual_only() -> None:
    assert (
        transition_provisional_to_approved(
            KnowledgeRecordStatus.provisional.value,
            promotion_source="manual_curator",
        )
        == KnowledgeRecordStatus.approved.value
    )


def test_transition_provisional_to_approved_rejects_ai_synthesis() -> None:
    with pytest.raises(KnowledgePromotionError, match="synthesis alone"):
        transition_provisional_to_approved(
            KnowledgeRecordStatus.provisional.value,
            promotion_source="ai_synthesis",
        )


def test_transition_provisional_to_approved_wrong_current_status() -> None:
    with pytest.raises(KnowledgePromotionError, match="provisional"):
        transition_provisional_to_approved(
            KnowledgeRecordStatus.needs_review.value,
            promotion_source="manual_curator",
        )


def test_transition_provisional_to_needs_review() -> None:
    assert (
        transition_provisional_to_needs_review(KnowledgeRecordStatus.provisional.value)
        == KnowledgeRecordStatus.needs_review.value
    )


def test_transition_provisional_to_needs_review_rejects_non_provisional() -> None:
    with pytest.raises(KnowledgePromotionError):
        transition_provisional_to_needs_review(KnowledgeRecordStatus.approved.value)


def test_transition_provisional_to_expired() -> None:
    assert (
        transition_provisional_to_expired(KnowledgeRecordStatus.provisional.value)
        == KnowledgeRecordStatus.expired.value
    )


def test_transition_provisional_to_expired_rejects_non_provisional() -> None:
    with pytest.raises(KnowledgePromotionError):
        transition_provisional_to_expired(KnowledgeRecordStatus.needs_review.value)


def test_transition_needs_review_to_approved_manual_only() -> None:
    assert (
        transition_needs_review_to_approved(
            KnowledgeRecordStatus.needs_review.value,
            promotion_source="manual_curator",
        )
        == KnowledgeRecordStatus.approved.value
    )


def test_transition_needs_review_to_approved_rejects_ai_synthesis() -> None:
    with pytest.raises(KnowledgePromotionError, match="synthesis alone"):
        transition_needs_review_to_approved(
            KnowledgeRecordStatus.needs_review.value,
            promotion_source="ai_synthesis",
        )


def test_is_medical_or_heuristic_gated_lane_true_for_medical_sensitivity() -> None:
    assert is_medical_or_heuristic_gated_lane(
        "medical",
        topic=None,
        species=None,
        life_stage=None,
        category=None,
    )


def test_is_medical_or_heuristic_gated_lane_true_for_cancer_heuristic() -> None:
    assert is_medical_or_heuristic_gated_lane(
        "general",
        topic="Cancer screening for pets",
        species=None,
        life_stage=None,
        category=None,
    )


def test_assert_sensitive_medical_bundle_policies_rejects_provisional_when_gated() -> None:
    with pytest.raises(KnowledgePromotionError, match="needs_review"):
        assert_sensitive_medical_bundle_policies(
            ingest_initial_status=KnowledgeRecordStatus.provisional.value,
            content_sensitivity="medical",
            topic=None,
            species=None,
            life_stage=None,
            category=None,
        )


def test_assert_sensitive_medical_bundle_policies_accepts_needs_review_when_gated() -> None:
    assert_sensitive_medical_bundle_policies(
        ingest_initial_status=KnowledgeRecordStatus.needs_review.value,
        content_sensitivity="medical",
        topic=None,
        species=None,
        life_stage=None,
        category=None,
    )


def test_transition_needs_review_to_approved_wrong_current() -> None:
    with pytest.raises(KnowledgePromotionError, match="needs_review"):
        transition_needs_review_to_approved(
            KnowledgeRecordStatus.provisional.value,
            promotion_source="manual_curator",
        )
