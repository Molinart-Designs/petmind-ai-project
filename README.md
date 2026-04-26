# PetMind AI — Personalized Pet Care Advisor API

> AI/LLM backend with a RAG architecture for personalized pet care guidance, deployed on AWS and designed to answer pet care questions using grounded document context and structured pet profile data.

---

## Production URL

**Base URL:** `https://api.petronum.ai`

### Public endpoints

- `GET /api/v1/health`
- `POST /api/v1/query`
- `POST /api/v1/ingest`

> Replace the base URL above with the current public ECS/Fargate endpoint or public IP used for the E3 submission.

---

## Project overview

PetMind AI is the academic backend deliverable for an **AI-LLM Solution Architect** course project.

The system implements:

- a FastAPI-based backend
- a Retrieval-Augmented Generation (RAG) pipeline
- PostgreSQL + pgvector as vector store
- OpenAI for embeddings and response generation
- AWS deployment using RDS, ECR, ECS Fargate, and CloudWatch Logs

The backend is designed to:

- ingest curated pet care knowledge
- answer natural-language pet care questions
- personalize responses using structured pet profile data
- provide grounded answers with confidence, sources, and safety disclaimers
- avoid definitive veterinary diagnoses

---

## Course conventions (do not break E2)

| Artifact                                                  | Canonical location                              |
| --------------------------------------------------------- | ----------------------------------------------- |
| Main project documentation                                | `docs/PROJECT_DOCUMENTATION.md`                 |
| Trusted external fallback design (L2/L3, flags, policies) | `docs/trusted-external-fallback.md`             |
| OpenAPI                                                   | `docs/api/openapi.yaml`                         |
| ADRs                                                      | `docs/adr/ADR-00x.md` + `docs/adr/ADR-00x-*.md` |

---

## Repository structure

```text
ProyectoAI/
├── .github/workflows/
│   ├── ci.yml
│   └── cd.yml
├── alembic/
├── alembic.ini
├── data/
│   ├── raw/
│   ├── processed/
│   └── curated/
├── docs/
│   ├── PROJECT_DOCUMENTATION.md
│   ├── trusted-external-fallback.md
│   ├── adr/
│   ├── api/openapi.yaml
│   ├── architecture/
│   ├── diagrams/
│   ├── threat-model.md
│   ├── ethics-and-compliance.md
│   └── deployment-guide.md
├── notebooks/
│   └── rag_evaluation.ipynb
├── reports/
├── scripts/
├── src/
│   ├── api/
│   ├── core/
│   ├── db/
│   ├── rag/
│   ├── research/
│   ├── security/
│   └── utils/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── load/
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pytest.ini
└── requirements.txt
```

---

## Main features

- POST /api/v1/query for grounded AI responses
- POST /api/v1/ingest for curated knowledge ingestion
- GET /api/v1/health for service and database health
- RAG pipeline with PostgreSQL + pgvector
- source-aware responses
- confidence scoring (high, medium, low)
- veterinary follow-up flag for sensitive scenarios
- safety guardrails against definitive clinical claims
- optional trusted external fallback architecture behind feature flags

## Technology stack

### Backend

- Python 3.11
- FastAPI
- SQLAlchemy
- Alembic

### AI / RAG

- OpenAI
- PostgreSQL
- pgvector

### Local development

- Docker
- Docker Compose

### Cloud deployment

- AWS RDS for PostgreSQL
- AWS ECR
- AWS ECS with Fargate
- AWS CloudWatch Logs

---

## Local development

### Run with Docker Compose

```
docker compose up --build
```

### Health check

```
curl http://127.0.0.1:8000/api/v1/health
```

### Interactive API docs

```
http://127.0.0.1:8000/docs
```

---

## Useful commands

```
make up
make down
make restart
make logs
make ps
make build
make shell
make db-shell
make test
make test-unit
make test-integration
make coverage
make coverage-html
make load
make ingest
make health
make clean
```

---

### Example production validation

## Health

```
curl http://<TU_PUBLIC_IP_O_URL>:8000/api/v1/health
```

## Query

I'm not showing secrets and sensitive information

```
curl -X POST http://<PUBLIC_IP>:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{
    "question": "What is a good feeding routine for my adult dog?",
    "pet_profile": {
      "species": "dog",
      "life_stage": "adult"
    },
    "filters": {
      "category": "nutrition",
      "species": "dog",
      "life_stage": "adult"
    },
    "top_k": 4
  }'
```

## Ingest

I'm not showing secrets and sensitive information

```
curl -X POST http://PUBLIC_IP:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: API_KEY" \
  -d @data/curated/sample_ingest.json
```

---

## Testing and quality

### The project includes:

- unit tests
- integration tests
- a real RAG end-to-end integration test
- coverage reporting with pytest-cov
- CI with GitHub Actions

### Current coverage

- 82.91%

### Coverage artifact

- reports/coverage.xml

---

## Deliverable status

- E1: scope and requirements documented
- E2: architecture, ADRs, OpenAPI, threat model, ethics, sections 3–5 completed
- E3: functional backend implemented, deployed on AWS, tested, with coverage and end-to-end RAG validation

---

## Documentation

Main project documentation:

- docs/PROJECT_DOCUMENTATION.md

Trusted external fallback architecture:

- docs/trusted-external-fallback.md

OpenAPI:

- docs/api/openapi.yaml

Deployment guidance:

- docs/deployment-guide.md

---

## Safety note

PetMind AI is designed to provide educational and grounded guidance, not definitive veterinary diagnosis.

The system:

- uses retrieved context as the basis for answers
- returns source traces when applicable
- lowers confidence when context is weak
- recommends veterinary follow-up in sensitive scenarios

---

## Diagrams

- docs/diagrams/c4-context.svg
- docs/diagrams/c4-container.svg
- docs/diagrams/data-flow.svg

Legacy E2 files:

- docs/architecture/architecture_general_es.svg
- docs/architecture/data_flow_es.svg

---

### Evaluation artifacts

- `reports/coverage.xml`
- `reports/ragas_report.json`
- `reports/load_test_summary.md`
