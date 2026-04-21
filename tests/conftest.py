from collections.abc import Generator
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.config import settings
from src.core.orchestrator import get_orchestrator


class FakeOrchestrator:
    async def answer(self, payload):
        return {
            "answer": (
                "For most healthy adult dogs, a consistent feeding schedule, "
                "fresh water, and portion control are good starting points."
            ),
            "needs_vet_followup": False,
            "confidence": "high",
            "sources": [
                {
                    "document_id": "doc-demo-1",
                    "chunk_id": "doc-demo-1-chunk-1",
                    "title": "Feeding basics for adult dogs",
                    "source": "curated-demo-source",
                    "category": "nutrition",
                    "species": "dog",
                    "life_stage": "adult",
                    "similarity_score": 0.92,
                    "snippet": "Adult dogs generally benefit from routine feeding schedules...",
                    "metadata": {
                        "source_batch": "demo_batch",
                        "chunk_index": 1,
                    },
                }
            ],
            "retrieval_count": 1,
            "used_filters": {
                "species": "dog",
                "life_stage": "adult",
            },
            "disclaimers": [
                "PetMind AI provides educational guidance based on curated information and does not replace professional veterinary evaluation."
            ],
            "generated_at": datetime.now(timezone.utc),
        }

    async def ingest(self, payload):
        return {
            "status": "completed",
            "source": payload.source,
            "documents_received": len(payload.documents),
            "documents_processed": len(payload.documents),
            "chunks_created": 3,
            "document_ids": ["doc-demo-1"],
            "message": "Documents ingested successfully into the knowledge base.",
            "ingested_at": datetime.now(timezone.utc),
        }


@pytest.fixture
def fake_orchestrator() -> FakeOrchestrator:
    return FakeOrchestrator()


@pytest.fixture
def client(fake_orchestrator: FakeOrchestrator) -> Generator[TestClient, None, None]:
    app.dependency_overrides[get_orchestrator] = lambda: fake_orchestrator
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {settings.api_key_header_name: settings.api_key}


@pytest.fixture
def sample_query_payload() -> dict:
    return {
        "question": "What is a good feeding routine for my adult dog?",
        "pet_profile": {
            "species": "dog",
            "breed": "Labrador",
            "age_years": 4,
            "life_stage": "adult",
            "weight_kg": 28,
            "sex": "male",
            "neutered": True,
            "conditions": [],
            "notes": "Very active dog",
        },
        "filters": {
            "category": "nutrition",
            "species": "dog",
            "life_stage": "adult",
        },
        "top_k": 4,
    }


@pytest.fixture
def sample_ingest_payload() -> dict:
    return {
        "source": "demo_batch",
        "documents": [
            {
                "external_id": "ext-001",
                "title": "Healthy feeding basics for adult dogs",
                "content": (
                    "Adult dogs benefit from routine feeding schedules, portion control, "
                    "fresh water access, and food choices appropriate to their age and activity level."
                ),
                "category": "nutrition",
                "species": "dog",
                "life_stage": "adult",
                "source_url": "https://example.com/dog-feeding-basics",
                "tags": ["feeding", "routine", "adult-dog"],
                "metadata": {
                    "reviewed_by": "petmind-demo",
                    "language": "en",
                },
            }
        ],
    }