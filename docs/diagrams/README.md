# Diagramas de arquitectura

| Archivo | Uso |
| ------- | --- |
| `c4-context.svg` / `.png` | C4 nivel **contexto** (sistema y actores externos). |
| `c4-container.svg` / `.png` | C4 nivel **contenedor** (API, DB, vector store, LLM, etc.). |
| `data-flow.svg` / `.png` | Flujo request → RAG → LLM → respuesta. |

**Nota:** Hoy `c4-container.svg` es una copia provisional del contexto hasta que dibujes el diagrama de contenedores definitivo en Draw.io u otra herramienta.

Las rutas `docs/architecture/*_es.svg` se mantienen como **legado** compatible con E2.
