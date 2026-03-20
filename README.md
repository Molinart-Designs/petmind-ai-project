# PetMind AI — Personalized Pet Care Advisor API

> API AI/LLM con arquitectura RAG desplegable en AWS, diseñada para responder consultas sobre cuidado de mascotas con contexto documental y personalización basada en el perfil de la mascota.

---

## 📌 Descripción

**PetMind AI** es el backend inteligente de una plataforma de acompañamiento para dueños de mascotas. Su objetivo es centralizar conocimiento curado sobre nutrición, comportamiento, rutinas y cuidados generales, y convertirlo en respuestas útiles, claras y personalizadas mediante un sistema **AI/LLM + RAG**.

En esta versión del proyecto, el foco está en el **backend evaluable del curso**: una API REST que recibe consultas en lenguaje natural, recupera contexto relevante desde una base de conocimiento y genera respuestas con fuentes asociadas. Aunque el producto real contempla una app móvil React Native, este repositorio se concentra en el componente AI/LLM desplegado en la nube y listo para integrarse con clientes web o mobile.

---

## 🏗️ Arquitectura

La solución sigue una arquitectura **RAG (Retrieval-Augmented Generation)**. El flujo general es:

1. El cliente envía una consulta al endpoint `/api/v1/query`
2. La API valida la solicitud y el contexto de la mascota
3. El motor RAG recupera documentos relevantes desde el vector store
4. El orquestador construye el prompt con contexto recuperado
5. El modelo LLM genera una respuesta controlada y segura
6. La API devuelve la respuesta junto con fuentes, latencia y metadatos

### Componentes principales

- **FastAPI** para la API REST
- **OpenAI** como proveedor LLM
- **LangChain o LlamaIndex** para orquestación
- **PostgreSQL + pgvector** o **ChromaDB** para recuperación semántica
- **Docker / Docker Compose** para entorno local reproducible
- **AWS** para despliegue cloud
- **GitHub Actions** para CI/CD

### Diagrama

Coloca aquí tu diagrama cuando lo tengas listo:

````md
## Estructura creada

```text
ProyectoAI/
├── README.md
├── .env.example
├── .gitignore
├── Makefile
├── requirements.txt
├── REQUIRED_FILES.md
├── docs/
│   └── PROJECT_DOCUMENTATION.md
├── src/
│   ├── __init__.py
│   ├── api/
│   │   └── __init__.py
│   ├── core/
│   │   └── __init__.py
│   ├── rag/
│   │   └── __init__.py
│   └── security/
│       └── __init__.py
└── tests/
    └── __init__.py
```
````

## Comandos utiles

```bash
make check-structure
make checklist-e1
```

## Siguiente paso (Semana 3-4)

- Diagrama C4.
- Diagrama de flujo de datos.
- ADR-001 y ADR-002.
- OpenAPI inicial.
