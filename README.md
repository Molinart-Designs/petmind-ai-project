# PetMind AI — Personalized Pet Care Advisor API

> API AI/LLM con arquitectura RAG desplegable en AWS, diseñada para responder consultas sobre cuidado de mascotas con contexto documental y personalización basada en el perfil de la mascota.

---

## Convenciones del curso (no romper E2)

| Artefacto | Ubicación canónica |
| --------- | ------------------- |
| Documentación principal | `docs/PROJECT_DOCUMENTATION.md` |
| OpenAPI | `docs/api/openapi.yaml` (única fuente) |
| ADRs (índice + contenido) | `docs/adr/ADR-00x.md` + `docs/adr/ADR-00x-*.md` |

---

## Estructura del repositorio

```text
ProyectoAI/
├── .github/workflows/
│   ├── ci.yml
│   └── cd.yml
├── alembic/                 # migraciones (E3+)
├── alembic.ini
├── data/
│   ├── raw/
│   ├── processed/
│   └── curated/
├── docs/
│   ├── PROJECT_DOCUMENTATION.md
│   ├── adr/
│   ├── api/openapi.yaml
│   ├── architecture/        # diagramas legado E2
│   ├── diagrams/            # C4 + flujo (recomendado)
│   ├── threat-model.md
│   ├── ethics-and-compliance.md
│   └── deployment-guide.md
├── notebooks/rag_evaluation.ipynb
├── scripts/
├── src/
│   ├── api/   (main, routes, schemas)
│   ├── core/
│   ├── db/
│   ├── rag/
│   ├── security/
│   └── utils/
├── tests/   (unit, integration, load)
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pytest.ini
└── requirements.txt
```

---

## Comandos útiles

```bash
make check-structure
make checklist-e2
make test
uvicorn src.api.main:app --reload --port 8000
# Docs interactivas: http://127.0.0.1:8000/docs
```

Docker (cuando tengas `.env`):

```bash
docker compose up --build
```

---

## Estado por entregable

- **E1:** alcance y requerimientos en `docs/PROJECT_DOCUMENTATION.md`.
- **E2:** diagramas (`docs/diagrams/` o legado `docs/architecture/`), ADRs, OpenAPI, secciones 3–5 del documento; `make checklist-e2` en verde.
- **E3:** implementar RAG completo, auth, CI/CD real, `cd.yml` y despliegue AWS.

---

## Diagramas

- **Nueva convención:** `docs/diagrams/c4-context.svg`, `c4-container.svg`, `data-flow.svg` (+ `.png` para entrega).
- **Legado E2:** `docs/architecture/architecture_general_es.svg`, `data_flow_es.svg`.

`c4-container.svg` es hoy una copia provisional del contexto; sustitúyelo por el diagrama de contenedores definitivo cuando lo tengas.
