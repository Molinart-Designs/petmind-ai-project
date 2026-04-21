from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from src.api.routes import router as api_router
from src.core.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting PetMind AI API",
        extra={
            "project_name": settings.project_name,
            "environment": settings.environment,
            "version": settings.app_version,
        },
    )
    yield
    logger.info("Shutting down PetMind AI API")


app = FastAPI(
    title="PetMind AI - Personalized Pet Care Advisor API",
    description=(
        "AI/LLM backend with RAG for personalized pet care guidance. "
        "This API provides health checks, knowledge ingestion, and natural language querying."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.api_prefix}/openapi.json",
)

app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")