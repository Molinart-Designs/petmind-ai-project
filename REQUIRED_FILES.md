# REQUIRED_FILES — E1 + E2 + estructura recomendada

## Curso (canónico)

- [x] `README.md`
- [x] `.env.example`
- [x] `.gitignore`
- [x] `Makefile`
- [x] `requirements.txt`
- [x] `docs/PROJECT_DOCUMENTATION.md`
- [x] `docs/api/openapi.yaml`
- [x] `docs/adr/ADR-001.md` (índice → contenido largo)
- [x] `docs/adr/ADR-002.md`
- [x] `docs/adr/ADR-003.md`
- [x] `docs/adr/ADR-001-openai-as-llm-base.md`
- [x] `docs/adr/ADR-002-postgresql-pgvector-as-vector-store.md`
- [x] `docs/adr/ADR-003-fastapi-as-framework.md`

## Diagramas (E2 — una de dos rutas)

- [x] Legado: `docs/architecture/*_es.svg` **o**
- [x] Recomendado: `docs/diagrams/c4-context.svg`, `c4-container.svg`, `data-flow.svg`

## Estructura extendida (pre-E3)

- [x] `.github/workflows/ci.yml` + `cd.yml` (CD placeholder)
- [x] `Dockerfile`, `docker-compose.yml`, `.dockerignore`
- [x] `pytest.ini`, `tests/unit`, `tests/integration`, `tests/load`
- [x] `notebooks/rag_evaluation.ipynb`
- [x] `docs/threat-model.md`, `docs/ethics-and-compliance.md`, `docs/deployment-guide.md`
- [x] `alembic/` + `alembic.ini`

## Verificación automática

```bash
make checklist-e2
make test
```
