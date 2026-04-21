# Modelo de amenazas (resumen)

Este documento resume el modelo de amenazas de PetMind AI. La versión detallada con tablas y controles está en **`docs/PROJECT_DOCUMENTATION.md`**, sección **5.1 Modelo de Amenazas y Controles de Seguridad**.

## Superficies principales

- API REST (`/query`, `/ingest`, `/health`)
- Pipeline RAG y corpus documental
- Proveedor LLM externo (OpenAI)
- Almacenamiento (PostgreSQL + pgvector)

## Controles prioritarios (v1)

1. API Key en endpoints protegidos (`X-API-Key`).
2. Rate limiting y validación de entrada.
3. Prompt restrictivo y límites de alcance (no diagnóstico veterinario).
4. RAG con umbral de similitud y corpus curado.
5. Secretos fuera del repositorio (`.env`, Secrets Manager en producción).
