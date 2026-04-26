"""Ensure trusted-research ORM tables are registered on Base.metadata."""


def test_research_persistence_tables_exist() -> None:
    import src.db.models  # noqa: F401 — register models on Base.metadata
    from src.db.session import Base

    names = {t.name for t in Base.metadata.sorted_tables}
    expected = {
        "knowledge_sources",
        "research_candidates",
        "research_candidate_sources",
        "knowledge_refresh_jobs",
    }
    assert expected.issubset(names)
    assert "document_chunks" in names
