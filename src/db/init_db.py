from sqlalchemy import text

from src.db.models import (  # noqa: F401
    DocumentChunk,
    KnowledgeRefreshJobRow,
    KnowledgeSource,
    Pet,
    QueryHistory,
    ResearchCandidate,
    ResearchCandidateSource,
    User,
)
from src.db.session import Base, engine
from src.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_pgvector_extension() -> None:
    """
    Ensures the pgvector extension exists in the current PostgreSQL database.
    """
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("Verified pgvector extension is available")
    except Exception as exc:
        logger.exception("Failed to ensure pgvector extension", extra={"error": str(exc)})
        raise


def create_tables() -> None:
    """
    Creates database tables declared in SQLAlchemy models.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as exc:
        logger.exception("Failed to create database tables", extra={"error": str(exc)})
        raise


def check_database_connection() -> None:
    """
    Executes a simple connectivity test against the database.
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Database connection check passed")
    except Exception as exc:
        logger.exception("Database connection check failed", extra={"error": str(exc)})
        raise


def init_db() -> None:
    """
    Full local database initialization flow.
    """
    logger.info("Initializing database")
    check_database_connection()
    ensure_pgvector_extension()
    create_tables()
    logger.info("Database initialization completed successfully")


if __name__ == "__main__":
    init_db()