# Guía de despliegue

Esta guía se completará en **E3** (contenedores, variables de AWS, pipeline CD).

## Estado actual (E2)

- Entorno local: ver **`README.md`** (`uvicorn`, Docker pendiente de afinar en E3).
- Especificación de API: **`docs/api/openapi.yaml`** (única fuente).

## Próximos pasos (E3)

1. Imagen Docker de la API y `docker-compose` con PostgreSQL + extensiones.
2. Workflow `cd.yml` con despliegue a AWS (ECS, App Runner u otro servicio acordado).
3. Secretos en GitHub Actions y en la nube (Secrets Manager).
