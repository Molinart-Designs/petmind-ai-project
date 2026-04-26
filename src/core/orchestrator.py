from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Literal, Protocol

from src.api.schemas import IngestRequest, QueryRequest
from src.core.config import settings
from src.core.llm_client import OpenAILLMClient, get_llm_client
from src.rag.ingestion import IngestionService, get_ingestion_service
from src.rag.retriever import Retriever, get_retriever
from src.research.ingest_candidates import (
    ExternalResearchIngestInput,
    ExternalResearchPersistencePort,
    NullExternalResearchPersistence,
    SqlAlchemyResearchCandidateStore,
    ingest_external_research_candidate,
)
from src.research.domain_authority import (
    ANCHOR_AUTHORITY_THRESHOLD_FOR_REDDIT_MIX,
    is_anecdotal_social_domain,
)
from src.research.http_url_utils import domain_from_http_url
from src.research.evidence_extractor import EvidenceExtractionResult
from src.research.evidence_quality import evidence_bundle_eligible_for_persistence
from src.research.schemas import ResearchResult
from src.research.trusted_research_service import (
    TrustedResearchOutcome,
    TrustedResearchRequest,
    get_disabled_trusted_research_service,
    get_trusted_research_service,
)
from src.security.guardrails import (
    EXTERNAL_PROVISIONAL_CONTEXT_DISCLAIMER,
    assess_query_risk,
    assess_retrieval_grounding,
    build_safe_fallback_answer,
    postprocess_answer,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _chunks_for_llm_empty_reason(
    *,
    internal_chunk_count: int,
    chunks_for_llm_count: int,
    will_run_l2: bool,
    used_external: bool,
    trigger_decision: str,
    research_outcome: TrustedResearchOutcome | None,
) -> str:
    """
    Machine-readable diagnosis when ``chunks_for_llm`` is empty (no LLM context → API fallback).

    Does not log by itself; used in structured ``query_context_resolution`` logs.
    """
    if chunks_for_llm_count > 0:
        return "not_empty"
    if internal_chunk_count > 0:
        return "invariant_broken_internal_nonempty_but_chunks_for_llm_empty"
    if not will_run_l2:
        return f"layer2_skipped:{trigger_decision}"
    if research_outcome is None:
        return "layer2_bug_no_research_outcome"
    if not used_external:
        detail = (research_outcome.debug_metrics or {}).get("external_failure_reason")
        if detail:
            return f"layer2_no_usable_snippets:{detail}"
        n = len(research_outcome.research.evidence.snippets)
        return f"layer2_no_usable_snippets:snippet_count={n}"
    return "invariant_broken_used_external_true_but_chunks_for_llm_empty"


def _firecrawl_configured_for_layer2() -> bool:
    return (
        settings.enable_trusted_external_retrieval
        and settings.trusted_content_provider.strip().lower() == "firecrawl"
        and bool(settings.firecrawl_api_key.strip())
    )


class SupportsTrustedResearch(Protocol):
    async def run(self, request: TrustedResearchRequest) -> TrustedResearchOutcome:
        """Trusted external research (Layer 2)."""


class RAGOrchestrator:
    def __init__(
        self,
        *,
        llm_client: OpenAILLMClient,
        retriever: Retriever,
        ingestion_service: IngestionService,
        trusted_research: SupportsTrustedResearch | None = None,
        external_research_store: ExternalResearchPersistencePort | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._retriever = retriever
        self._ingestion_service = ingestion_service
        if trusted_research is not None:
            self._trusted_research = trusted_research
        elif settings.enable_trusted_external_retrieval and settings.allow_provisional_in_query:
            self._trusted_research = get_trusted_research_service()
        else:
            self._trusted_research = get_disabled_trusted_research_service()
        self._external_store = (
            external_research_store
            if external_research_store is not None
            else NullExternalResearchPersistence()
        )

    @staticmethod
    def _internal_context_reasonably_covers_question(*, grounding: dict[str, Any]) -> bool:
        """
        El RAG interno tiene prioridad cuando hay contexto recuperado y “suficientemente fuerte”:

        - ``retrieval_count`` > 0 (hay chunks internos),
        - ``matched_count`` > 0 (al menos un chunk ≥ ``SIMILARITY_THRESHOLD``),
        - ``top_score`` ≥ ``EXTERNAL_RESEARCH_TRIGGER_THRESHOLD`` (el mejor score interno alcanza
          la barra mínima para no pedir refuerzo L2).

        Si esto es verdadero, **no** se dispara investigación externa (no hay ruta alternativa silenciosa).
        """
        if not grounding.get("has_any_context"):
            return False
        if grounding.get("retrieval_count", 0) == 0:
            return False
        if grounding.get("matched_count", 0) == 0:
            return False
        top = grounding.get("top_score")
        if top is None:
            return False
        return float(top) >= float(settings.external_research_trigger_threshold)

    def _should_trigger_trusted_external(
        self,
        *,
        grounding: dict[str, Any],
    ) -> bool:
        if not settings.enable_trusted_external_retrieval:
            return False
        if not settings.allow_provisional_in_query:
            return False
        return not self._internal_context_reasonably_covers_question(grounding=grounding)

    def _trusted_external_trigger_decision(
        self,
        *,
        grounding: dict[str, Any],
        risk_sensitive: bool,
    ) -> str:
        """Machine-readable reason for L2 (never log secrets or raw questions beyond preview elsewhere)."""
        if risk_sensitive:
            return "blocked_sensitive_query"
        if not settings.enable_trusted_external_retrieval:
            return "disabled_enable_trusted_external_retrieval"
        if not settings.allow_provisional_in_query:
            return "disabled_allow_provisional_in_query"
        if self._internal_context_reasonably_covers_question(grounding=grounding):
            return "skip_internal_context_sufficient"
        return "trigger_external_fallback"

    def _log_trusted_external_query_diag(
        self,
        *,
        grounding: dict[str, Any],
        trigger_decision: str,
        will_run_l2: bool,
        research_outcome: TrustedResearchOutcome | None,
        used_external: bool,
        answer_source: str | None,
        knowledge_status: str | None,
        is_fallback: bool,
    ) -> None:
        """Structured runtime observability for internal vs Layer-2 trusted external (no secrets)."""
        extm = (research_outcome.debug_metrics or {}) if research_outcome else {}
        extra: dict[str, Any] = {
            "event": "trusted_external_query_diag",
            "is_fallback_response": is_fallback,
            "internal_retrieval_count": grounding.get("retrieval_count"),
            "internal_matched_count": grounding.get("matched_count"),
            "internal_top_score": grounding.get("top_score"),
            "internal_has_any_context": grounding.get("has_any_context"),
            "internal_has_sufficient_context": grounding.get("has_sufficient_context"),
            "internal_similarity_threshold": float(settings.similarity_threshold),
            "external_research_trigger_threshold": float(settings.external_research_trigger_threshold),
            "external_trigger_decision": trigger_decision,
            "will_run_layer2": will_run_l2,
            "trusted_research_service_class": type(self._trusted_research).__name__,
            "used_external": used_external,
            "answer_source": answer_source,
            "knowledge_status": knowledge_status,
            "external_search_result_count": extm.get("external_search_result_count"),
            "external_allowlisted_hit_count": extm.get("external_allowlisted_hit_count"),
            "external_policy_blocked_url_count": extm.get("external_policy_blocked_url_count"),
            "external_ranked_unique_url_count": extm.get("external_ranked_unique_url_count"),
            "extracted_snippet_count": extm.get("extracted_snippet_count"),
            "firecrawl_pages_fetched": extm.get("firecrawl_pages_fetched"),
            "firecrawl_pages_substantial_body": extm.get("firecrawl_pages_substantial_body"),
            "layer2_search_provider_id": extm.get("search_provider_id"),
            "external_failure_reason": extm.get("external_failure_reason"),
        }
        if is_fallback:
            logger.warning("trusted_external_query_diag", extra=extra)
        else:
            logger.info("trusted_external_query_diag", extra=extra)

    @staticmethod
    def _merge_internal_then_external(
        internal: list[dict[str, Any]],
        external: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [*internal, *external]

    @staticmethod
    def _research_result_to_llm_chunks(
        research: ResearchResult,
        extraction: EvidenceExtractionResult | None = None,
    ) -> list[dict[str, Any]]:
        """Shape trusted retrieval snippets for LLM context (evidence only; never synthesis text)."""
        url_by_snippet: dict[str, str] = {}
        title_by_snippet: dict[str, str] = {}
        if extraction is not None:
            for c in extraction.frontend.citations:
                url_by_snippet[c.snippet_id] = str(c.source_url)
                st = (c.source_title or "").strip()
                if st:
                    title_by_snippet[c.snippet_id] = st
        sources_by_id = {s.id: s for s in research.evidence.sources}
        rows: list[dict[str, Any]] = []
        for sn in research.evidence.snippets:
            src = sources_by_id.get(sn.external_source_id)
            fallback_url = str(src.base_url) if src else ""
            source_url = url_by_snippet.get(sn.id) or fallback_url
            domain = domain_from_http_url(source_url)
            title = (
                (title_by_snippet.get(sn.id) or "").strip()
                or (sn.page_title or "").strip()
                or "Trusted external excerpt (provisional)"
            )
            rank = float(sn.ranking_score) if sn.ranking_score is not None else float(sn.authority_score)
            is_reddit = is_anecdotal_social_domain(domain)
            trusted_anchor = (not is_reddit) and float(sn.authority_score) >= ANCHOR_AUTHORITY_THRESHOLD_FOR_REDDIT_MIX
            rows.append(
                {
                    "chunk_id": sn.id,
                    "document_id": sn.external_source_id,
                    "title": title,
                    "source": f"trusted_external:{sn.id[:16]}",
                    "category": None,
                    "species": sn.species,
                    "breed": sn.breed,
                    "life_stage": sn.life_stage,
                    "similarity_score": rank,
                    "snippet": sn.text,
                    "metadata": {
                        "provisional": True,
                        "layer": "trusted_external",
                        "external_source_id": sn.external_source_id,
                        "source_url": source_url,
                        "source_domain": domain,
                        "ranking_score": sn.ranking_score,
                        "source_authority": sn.authority_score,
                        "anecdotal_supplemental": is_reddit,
                        "trusted_evidence_anchor": trusted_anchor,
                    },
                }
            )
        return rows

    @staticmethod
    def _snippet_preview(chunk: dict[str, Any], max_len: int = 240) -> str:
        raw = (
            chunk.get("snippet")
            or chunk.get("content")
            or chunk.get("text")
            or ""
        ).strip()
        if len(raw) <= max_len:
            return raw
        return raw[: max_len - 1].rstrip() + "…"

    def _build_evidence_catalog_for_review(self, chunks: list[dict[str, Any]]) -> str:
        """
        Índice estructurado para el prompt de ``review_draft`` (solo revisores).

        Se construye **únicamente** a partir de ``chunks`` ya materializados (internos + externos
        vía ``_research_result_to_llm_chunks``): URLs/dominios salen de metadatos de recuperación /
        citas de extracción, **nunca** del texto de la respuesta LLM ni del borrador de síntesis.
        """
        lines: list[str] = []
        for i, ch in enumerate(chunks, start=1):
            meta = ch.get("metadata") if isinstance(ch.get("metadata"), dict) else {}
            url = meta.get("source_url") or meta.get("url") or ""
            dom = meta.get("source_domain") or ""
            cid = ch.get("chunk_id") or ""
            did = ch.get("document_id") or ""
            src = ch.get("source") or ""
            title = ch.get("title") or ""
            preview = self._snippet_preview(ch)
            prov = meta.get("provisional") is True
            layer = "PROVISIONAL_EXTERNAL" if prov else "INTERNAL_KB"
            dom_line = f"  domain={dom}\n" if dom else ""
            lines.append(
                f"[E{i} | {layer}] chunk_id={cid} document_id={did}\n"
                f"  source={src} title={title}\n"
                f"  url={url if url else '(none)'}\n"
                f"{dom_line}"
                f"  excerpt: {preview}"
            )
        return "\n\n".join(lines)

    def _build_review_draft_system_prompt(self, *, is_sensitive: bool, is_medical: bool) -> str:
        base = (
            "You are an internal PetMind analyst writing for human editors and reviewers only. "
            "Never address a pet owner; never write marketing tone. "
            "Produce an article-style synthesis (sections with short headings) that organizes what the "
            "retrieved evidence supports. "
            "Every substantive claim must be traceable: cite chunk_id values and URLs from the evidence catalog "
            "where available, and quote or paraphrase only what the excerpts support. "
            "This draft is NOT a source of truth for clinical or legal decisions: raw evidence rows and "
            "curated internal KB supersede any narrative here. If evidence is thin or conflicting, say so explicitly. "
            "Include a final section 'Evidence index' listing each chunk_id you relied on."
        )
        if is_sensitive:
            base += (
                " The user question appears urgent or sensitive: keep the synthesis short, foreground "
                "limitations and escalation considerations, and do not imply definitive diagnosis or treatment."
            )
        elif is_medical:
            base += (
                " The topic is medical-adjacent: be explicit about uncertainty and the need for licensed "
                "veterinary judgment."
            )
        return base

    def _build_review_draft_user_prompt(
        self,
        *,
        question: str,
        pet_profile: dict[str, Any],
        evidence_catalog: str,
        merged_chunks: list[dict[str, Any]],
        external_evidence_summary: str | None = None,
    ) -> str:
        ctx = self._format_context(merged_chunks)
        pet_block = self._format_pet_profile(pet_profile)
        extra = ""
        if external_evidence_summary:
            extra = f"\nTrusted external retrieval summary:\n{external_evidence_summary}\n"
        return (
            f"User question (context only):\n{question}\n\n"
            f"Pet profile:\n{pet_block}\n"
            f"{extra}\n"
            "Evidence catalog (use these identifiers and URLs in your citations):\n"
            f"{evidence_catalog}\n\n"
            "Full knowledge context (same passages as shown to the user-facing model):\n"
            f"{ctx}\n"
        )

    async def answer(self, payload: QueryRequest) -> dict[str, Any]:
        question = payload.question.strip()
        if not question:
            raise ValueError("Question must not be empty.")

        risk = assess_query_risk(question)
        retrieval_filters = self._build_retrieval_filters(payload)
        top_k = payload.top_k or settings.retriever_top_k

        logger.info(
            "Starting RAG answer flow",
            extra={
                "top_k": top_k,
                "filters": retrieval_filters,
                "is_sensitive": risk.is_sensitive,
                "is_medical": risk.is_medical,
            },
        )

        retrieved_chunks = await self._retriever.retrieve(
            question=question,
            top_k=top_k,
            filters=retrieval_filters,
        )

        grounding = assess_retrieval_grounding(
            retrieved_chunks=retrieved_chunks,
            similarity_threshold=settings.similarity_threshold,
        )

        trigger_decision = self._trusted_external_trigger_decision(
            grounding=grounding,
            risk_sensitive=risk.is_sensitive,
        )
        will_run_l2 = self._should_trigger_trusted_external(grounding=grounding) and not risk.is_sensitive

        logger.debug(
            "trusted_external_fallback_grounding",
            extra={
                "event": "trusted_external_fallback",
                "phase": "internal_grounding",
                "internal_retrieval_count": grounding.get("retrieval_count"),
                "internal_matched_count": grounding.get("matched_count"),
                "internal_top_score": grounding.get("top_score"),
                "internal_has_sufficient_context": grounding.get("has_sufficient_context"),
                "external_trigger_decision": trigger_decision,
                "external_l2_will_run": will_run_l2,
                "trusted_research_service_class": type(self._trusted_research).__name__,
                "external_research_trigger_threshold": float(settings.external_research_trigger_threshold),
            },
        )

        research_outcome = None
        used_external = False
        external_evidence_gate_ok = True

        if will_run_l2:
            tr_req = TrustedResearchRequest(
                question=question,
                pet_profile=payload.pet_profile,
                filters=payload.filters,
            )
            research_outcome = await self._trusted_research.run(tr_req)
            if research_outcome.research.evidence.snippets:
                external_evidence_gate_ok, gate_reasons = evidence_bundle_eligible_for_persistence(
                    research_outcome.extraction
                )
                if external_evidence_gate_ok:
                    used_external = True
                    logger.info(
                        "Trusted external research augmented context",
                        extra={
                            "snippet_count": len(research_outcome.research.evidence.snippets),
                            "external_confidence": research_outcome.research.external_confidence,
                        },
                    )
                else:
                    logger.warning(
                        "external_evidence_quality_gate_failed_merge_skipped",
                        extra={"reasons": gate_reasons},
                    )

        internal_for_grounding = list(retrieved_chunks)
        chunks_for_llm = list(retrieved_chunks)
        if used_external and research_outcome is not None:
            ext_chunks = self._research_result_to_llm_chunks(
                research_outcome.research,
                research_outcome.extraction,
            )
            chunks_for_llm = self._merge_internal_then_external(retrieved_chunks, ext_chunks)

        ext_metrics: dict[str, Any] = {}
        if research_outcome is not None and research_outcome.debug_metrics:
            ext_metrics = dict(research_outcome.debug_metrics)

        should_trigger_trusted_external = self._should_trigger_trusted_external(grounding=grounding)
        chunks_empty_reason = _chunks_for_llm_empty_reason(
            internal_chunk_count=len(retrieved_chunks),
            chunks_for_llm_count=len(chunks_for_llm),
            will_run_l2=will_run_l2,
            used_external=used_external,
            trigger_decision=trigger_decision,
            research_outcome=research_outcome,
        )

        logger.info(
            "query_context_resolution",
            extra={
                "event": "query_context_resolution",
                "question_preview": question[:120],
                "internal_retrieved_chunks_count": len(retrieved_chunks),
                "internal_grounding_retrieval_count": grounding.get("retrieval_count"),
                "internal_grounding_matched_count": grounding.get("matched_count"),
                "internal_grounding_top_score": grounding.get("top_score"),
                "internal_similarity_threshold": float(settings.similarity_threshold),
                "external_research_trigger_threshold": float(settings.external_research_trigger_threshold),
                "should_trigger_trusted_external": should_trigger_trusted_external,
                "will_run_layer2_effective": will_run_l2,
                "is_sensitive_query": risk.is_sensitive,
                "trusted_research_service_class": type(self._trusted_research).__name__,
                "external_search_result_count": ext_metrics.get("external_search_result_count"),
                "external_allowlisted_hit_count": ext_metrics.get("external_allowlisted_hit_count"),
                "external_policy_blocked_url_count": ext_metrics.get("external_policy_blocked_url_count"),
                "extracted_snippet_count": ext_metrics.get("extracted_snippet_count"),
                "research_evidence_snippet_count": (
                    len(research_outcome.research.evidence.snippets) if research_outcome else 0
                ),
                "chunks_for_llm_count": len(chunks_for_llm),
                "chunks_for_llm_empty_reason": chunks_empty_reason,
                "used_external": used_external,
                "final_answer_source": ("fallback" if not chunks_for_llm else "llm_path_see_post_postprocess_log"),
                "final_knowledge_status": ("none" if not chunks_for_llm else "llm_path_see_post_postprocess_log"),
                "setting_enable_trusted_external_retrieval": settings.enable_trusted_external_retrieval,
                "setting_allow_provisional_in_query": settings.allow_provisional_in_query,
                "setting_trusted_search_provider": settings.trusted_search_provider.strip().lower(),
                "setting_tavily_api_key_present": bool(settings.tavily_api_key.strip()),
                "setting_trusted_external_allowlist_configured": bool(
                    settings.trusted_external_allowlist_domains.strip()
                ),
                "setting_trusted_content_provider": settings.trusted_content_provider.strip().lower(),
                "setting_firecrawl_api_key_present": bool(settings.firecrawl_api_key.strip()),
                "setting_firecrawl_would_activate_layer2_fetch": _firecrawl_configured_for_layer2(),
                "layer2_search_provider_id": ext_metrics.get("search_provider_id"),
                "layer2_external_failure_reason": ext_metrics.get("external_failure_reason"),
            },
        )

        if not chunks_for_llm:
            logger.warning(
                "No retrieval context found, returning fallback answer",
                extra={
                    "matched_count": grounding["matched_count"],
                    "top_score": grounding["top_score"],
                    "external_attempted": research_outcome is not None,
                    "external_failure_reason": (research_outcome.debug_metrics or {}).get(
                        "external_failure_reason"
                    )
                    if research_outcome
                    else None,
                    "chunks_for_llm_empty_reason": chunks_empty_reason,
                },
            )

            fallback = build_safe_fallback_answer(
                question=question,
                retrieved_chunks=retrieved_chunks,
            )
            fallback["used_filters"] = retrieval_filters
            fallback["generated_at"] = datetime.now(timezone.utc)
            self._log_trusted_external_query_diag(
                grounding=grounding,
                trigger_decision=trigger_decision,
                will_run_l2=will_run_l2,
                research_outcome=research_outcome,
                used_external=used_external,
                answer_source=fallback.get("answer_source"),
                knowledge_status=fallback.get("knowledge_status"),
                is_fallback=True,
            )
            return fallback

        frontend_system = self._build_frontend_system_prompt(
            risk.is_sensitive,
            risk.is_medical,
            provisional_external=used_external,
        )
        user_prompt = self._build_user_prompt(
            question=question,
            pet_profile=payload.pet_profile.model_dump(exclude_none=True)
            if payload.pet_profile
            else {},
            retrieved_chunks=chunks_for_llm,
        )

        llm_frontend_raw = await self._llm_client.generate_text(
            system_prompt=frontend_system,
            user_prompt=user_prompt,
            max_output_tokens=500 if risk.is_sensitive else 600,
        )

        result = postprocess_answer(
            answer=llm_frontend_raw,
            question=question,
            retrieved_chunks=chunks_for_llm,
            similarity_threshold=settings.similarity_threshold,
            chunks_for_grounding=internal_for_grounding if used_external else None,
            used_provisional_external=used_external,
        )

        if used_external:
            result["disclaimers"] = sorted(
                set(result["disclaimers"]) | {EXTERNAL_PROVISIONAL_CONTEXT_DISCLAIMER}
            )
            result["answer_source"] = "external_trusted"
            result["knowledge_status"] = "provisional"
        else:
            result["answer_source"] = "internal"
            result["knowledge_status"] = "approved"

        logger.info(
            "query_context_resolution",
            extra={
                "event": "query_context_resolution",
                "phase": "post_postprocess",
                "chunks_for_llm_count": len(chunks_for_llm),
                "final_answer_source": result.get("answer_source"),
                "final_knowledge_status": result.get("knowledge_status"),
                "used_external": used_external,
            },
        )

        evidence_catalog = self._build_evidence_catalog_for_review(chunks_for_llm)
        ext_summary = research_outcome.research.evidence_summary if research_outcome else None
        review_system = self._build_review_draft_system_prompt(
            is_sensitive=risk.is_sensitive,
            is_medical=risk.is_medical,
        )
        review_user = self._build_review_draft_user_prompt(
            question=question,
            pet_profile=payload.pet_profile.model_dump(exclude_none=True)
            if payload.pet_profile
            else {},
            evidence_catalog=evidence_catalog,
            merged_chunks=chunks_for_llm,
            external_evidence_summary=ext_summary if used_external else None,
        )
        review_max_tokens = 450 if risk.is_sensitive else 1200
        run_review_draft_llm = (not used_external) or external_evidence_gate_ok
        if run_review_draft_llm:
            review_draft_raw = await self._llm_client.generate_text(
                system_prompt=review_system,
                user_prompt=review_user,
                max_output_tokens=review_max_tokens,
            )
            result["review_draft"] = review_draft_raw.strip()
        else:
            result["review_draft"] = None

        provisional_candidate_saved = False
        if (
            used_external
            and external_evidence_gate_ok
            and research_outcome is not None
            and settings.enable_auto_save_provisional_knowledge
            and settings.enable_trusted_external_retrieval
            and settings.allow_provisional_in_query
        ):
            sens: Literal["general", "medical", "behavioral", "nutrition", "other"] = (
                "medical" if risk.is_medical else "general"
            )
            rd = result.get("review_draft") or ""
            inp = ExternalResearchIngestInput(
                extraction=research_outcome.extraction,
                research_result=research_outcome.research,
                content_sensitivity=sens,
                synthesis_text=None,
                frontend_answer_text=result["answer"],
                internal_review_llm_draft=rd or None,
            )
            try:
                await ingest_external_research_candidate(inp, store=self._external_store)
                provisional_candidate_saved = True
            except Exception:
                logger.exception(
                    "Failed to persist external research candidate",
                    extra={"question_preview": question[:120]},
                )

        result["used_filters"] = retrieval_filters
        result["generated_at"] = datetime.now(timezone.utc)

        layer2_metrics = (
            dict(research_outcome.debug_metrics)
            if research_outcome is not None and research_outcome.debug_metrics is not None
            else {}
        )

        logger.debug(
            "trusted_external_fallback_orchestrator_summary",
            extra={
                "event": "trusted_external_fallback",
                "phase": "orchestrator_summary",
                "internal_retrieval_count": grounding.get("retrieval_count"),
                "internal_top_score": grounding.get("top_score"),
                "external_trigger_decision": trigger_decision,
                "used_external": used_external,
                "provisional_candidate_saved": provisional_candidate_saved,
                **layer2_metrics,
            },
        )

        self._log_trusted_external_query_diag(
            grounding=grounding,
            trigger_decision=trigger_decision,
            will_run_l2=will_run_l2,
            research_outcome=research_outcome,
            used_external=used_external,
            answer_source=result.get("answer_source"),
            knowledge_status=result.get("knowledge_status"),
            is_fallback=False,
        )

        logger.info(
            "Completed RAG answer flow",
            extra={
                "retrieval_count": result["retrieval_count"],
                "confidence": result["confidence"],
                "needs_vet_followup": result["needs_vet_followup"],
                "used_external": used_external,
                "answer_source": result.get("answer_source"),
                "knowledge_status": result.get("knowledge_status"),
            },
        )

        return result

    async def ingest(self, payload: IngestRequest) -> dict[str, Any]:
        if not payload.documents:
            raise ValueError("At least one document is required for ingestion.")

        logger.info(
            "Starting ingestion flow",
            extra={
                "source": payload.source,
                "documents_received": len(payload.documents),
            },
        )

        result = await self._ingestion_service.ingest_documents(payload)

        response = {
            "status": "completed",
            "source": payload.source,
            "documents_received": len(payload.documents),
            "documents_processed": result["documents_processed"],
            "chunks_created": result["chunks_created"],
            "document_ids": result["document_ids"],
            "message": result.get(
                "message",
                "Documents ingested successfully into the knowledge base.",
            ),
            "ingested_at": datetime.now(timezone.utc),
        }

        logger.info(
            "Completed ingestion flow",
            extra={
                "source": payload.source,
                "documents_processed": response["documents_processed"],
                "chunks_created": response["chunks_created"],
            },
        )

        return response

    def _build_retrieval_filters(self, payload: QueryRequest) -> dict[str, Any]:
        filters: dict[str, Any] = {}

        if payload.filters:
            filters.update(payload.filters.model_dump(exclude_none=True))

        if payload.pet_profile:
            pet_profile = payload.pet_profile.model_dump(exclude_none=True)

            if "species" not in filters and pet_profile.get("species"):
                filters["species"] = pet_profile["species"]

            if "life_stage" not in filters and pet_profile.get("life_stage"):
                filters["life_stage"] = pet_profile["life_stage"]

        return filters

    def _build_frontend_system_prompt(
        self,
        is_sensitive: bool,
        is_medical: bool,
        *,
        provisional_external: bool = False,
    ) -> str:
        safety_block = (
            "You are PetMind AI, a pet care guidance assistant. "
            "You must answer only from the provided context. "
            "Do not invent facts. "
            "Do not provide definitive veterinary diagnoses. "
            "Do not prescribe medication dosages unless that exact information is explicitly present in the provided context, "
            "and even then, advise professional veterinary confirmation. "
            "If the context is insufficient, say so clearly. "
            "Prefer educational, cautious, practical guidance."
        )

        medical_block = (
            "If the question involves symptoms, treatment, diagnosis, medication, or health risk, "
            "make it clear that professional veterinary evaluation may be needed. "
            "Stay conservative: avoid treatment specifics not explicitly supported by the context."
            if is_medical
            else "If relevant, encourage safe monitoring and good pet care practices."
        )

        sensitive_block = (
            "This appears potentially urgent or sensitive. "
            "Advise prompt veterinary attention if symptoms are severe, sudden, or worsening. "
            "Keep the user-facing reply brief and safety-first; avoid speculative detail."
            if is_sensitive
            else "Keep the tone calm, clear, and grounded."
        )

        external_block = (
            "The knowledge context lists INTERNAL passages first (approved knowledge base), then optional "
            "PROVISIONAL trusted external excerpts. Prefer internal passages when they fully address the question. "
            "Treat external excerpts as supplementary; they are not independently verified as internal content. "
            "Never present external excerpts as definitive medical fact."
            if provisional_external
            else ""
        )

        style_block = (
            "OUTPUT: Produce only the concise, user-facing answer for the pet owner. "
            "Do not include internal review sections, evidence tables, or staff-only notes. "
            "Respond in concise natural language. "
            "Use short paragraphs or bullets only when helpful. "
            "Base the answer strictly on the provided knowledge context and pet profile."
        )

        parts = [safety_block, medical_block, sensitive_block, external_block, style_block]
        return "\n".join(p for p in parts if p)

    def _build_user_prompt(
        self,
        *,
        question: str,
        pet_profile: dict[str, Any],
        retrieved_chunks: list[dict[str, Any]],
    ) -> str:
        pet_profile_block = self._format_pet_profile(pet_profile)
        context_block = self._format_context(retrieved_chunks)

        return (
            f"User question:\n{question}\n\n"
            f"Pet profile:\n{pet_profile_block}\n\n"
            f"Knowledge context:\n{context_block}\n\n"
            "Instructions:\n"
            "- Answer using only the knowledge context above.\n"
            "- If the context does not fully support a claim, say that clearly.\n"
            "- Do not present speculative statements as facts.\n"
            "- Be helpful, practical, and safety-conscious.\n"
            "- Write as the direct reply to the pet owner (no internal-only commentary).\n"
        )

    def _format_pet_profile(self, pet_profile: dict[str, Any]) -> str:
        if not pet_profile:
            return "No pet profile provided."

        lines = []
        for key, value in pet_profile.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _format_context(self, retrieved_chunks: list[dict[str, Any]]) -> str:
        if not retrieved_chunks:
            return "No context retrieved."

        formatted_chunks: list[str] = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            title = chunk.get("title") or "Untitled"
            source = chunk.get("source") or "unknown"
            snippet = (
                chunk.get("snippet")
                or chunk.get("content")
                or chunk.get("text")
                or ""
            ).strip()
            similarity_score = chunk.get("similarity_score")
            meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
            provisional = meta.get("provisional") is True
            domain = meta.get("source_domain") or ""
            domain_line = f"Domain: {domain}" if domain else None
            section = (
                f"[Chunk {index}] — PROVISIONAL EXTERNAL (not internally approved)"
                if provisional
                else f"[Chunk {index}] — internal knowledge base"
            )

            body_lines = [
                section,
                f"Title: {title}",
                f"Source: {source}",
            ]
            if domain_line:
                body_lines.append(domain_line)
            body_lines.extend(
                [
                    f"Similarity: {similarity_score}",
                    f"Content: {snippet}",
                ]
            )
            formatted_chunks.append("\n".join(body_lines))

        return "\n\n".join(formatted_chunks)


@lru_cache
def get_orchestrator() -> RAGOrchestrator:
    # ``SqlAlchemyResearchCandidateStore`` solo se instancia si los tres flags están activos:
    # no abre conexión DB en ``__init__`` (solo guarda ``session_factory``); I/O al primer persist.
    store: ExternalResearchPersistencePort = (
        SqlAlchemyResearchCandidateStore()
        if (
            settings.enable_auto_save_provisional_knowledge
            and settings.enable_trusted_external_retrieval
            and settings.allow_provisional_in_query
        )
        else NullExternalResearchPersistence()
    )
    return RAGOrchestrator(
        llm_client=get_llm_client(),
        retriever=get_retriever(),
        ingestion_service=get_ingestion_service(),
        trusted_research=None,
        external_research_store=store,
    )
