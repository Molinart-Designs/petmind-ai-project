from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text

from src.api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    query_response_dict_from_orchestrator,
)
from src.core.config import settings
from src.core.orchestrator import RAGOrchestrator, get_orchestrator
from src.db.session import get_db_session
from src.security.auth import get_request_auth_for_ingest, get_request_auth_for_query
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["PetMind AI"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
)
async def health_check() -> HealthResponse:
    """
    Basic service health endpoint.
    Useful for local checks, Docker healthchecks, and later ECS/Fargate health probes.
    """
    db_status = "unknown"

    try:
        with get_db_session() as session:
            session.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as exc:
        db_status = "unhealthy"
        logger.exception("Database health check failed", extra={"error": str(exc)})

    overall_status = "healthy" if db_status == "healthy" else "degraded"

    return HealthResponse(
        status=overall_status,
        service=settings.project_name,
        version=settings.app_version,
        environment=settings.environment,
        database=db_status,
        timestamp=datetime.now(timezone.utc),
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query PetMind AI with RAG context",
    dependencies=[Depends(get_request_auth_for_query)],
)
async def query_petmind(
    payload: QueryRequest,
    orchestrator: RAGOrchestrator = Depends(get_orchestrator),
) -> QueryResponse:
    """
    Receives a natural-language question plus optional pet profile metadata,
    retrieves relevant context from the knowledge base, and generates a grounded answer.
    """
    try:
        logger.info(
            "Received query request",
            extra={
                "question_preview": payload.question[:120],
                "has_pet_profile": payload.pet_profile is not None,
                "top_k": payload.top_k or settings.retriever_top_k,
            },
        )

        result = await orchestrator.answer(payload)

        # Public API: never expose internal LLM review drafts over HTTP (field stays null for clients).
        result = {**result, "review_draft": None}
        public_payload = query_response_dict_from_orchestrator(result)
        public_payload["review_draft"] = None

        return QueryResponse.model_validate(public_payload)

    except ValueError as exc:
        logger.warning("Validation/business error in query flow", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Unhandled error while processing query", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the query.",
        ) from exc


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest curated pet care knowledge into the vector store",
    dependencies=[Depends(get_request_auth_for_ingest)],
)
async def ingest_documents(
    payload: IngestRequest,
    orchestrator: RAGOrchestrator = Depends(get_orchestrator),
) -> IngestResponse:
    """
    Ingests curated knowledge documents into the RAG pipeline.
    This endpoint is intended for controlled/admin use, not for public user traffic.
    """
    try:
        logger.info(
            "Received ingestion request",
            extra={
                "documents_count": len(payload.documents),
                "source": payload.source,
            },
        )

        result = await orchestrator.ingest(payload)

        return IngestResponse(**result)

    except ValueError as exc:
        logger.warning("Validation/business error in ingestion flow", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Unhandled error while processing ingestion", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while ingesting documents.",
        ) from exc