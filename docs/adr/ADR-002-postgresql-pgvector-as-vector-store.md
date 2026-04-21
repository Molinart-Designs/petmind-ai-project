# ADR-002: Usar PostgreSQL con pgvector como vector store principal

**Fecha:** 05/04/2026  
**Estado:** Aceptado  
**Autores:** Emilio Molina  
**Revisado por:** _Pendiente_

## Contexto

PetMind AI necesita un almacenamiento vectorial para soportar recuperación semántica sobre documentos curados relacionados con cuidado de mascotas. El sistema también requiere persistencia estructurada para datos operativos, como sesiones, documentos, metadatos y trazabilidad de consultas. Para la primera versión del proyecto, se busca minimizar complejidad operativa sin comprometer la escalabilidad futura.

La decisión debe balancear simplicidad de despliegue, integración con el backend y capacidad de evolución hacia un sistema más completo.

## Decisión

Se decide usar **PostgreSQL con pgvector** como vector store principal del sistema.

## Opciones Evaluadas

| Opción                    | Latencia (aprox.) | Costo/mes (aprox.) | Escalabilidad | Facilidad integración |
| ------------------------- | ----------------- | ------------------ | ------------- | --------------------- |
| **PostgreSQL + pgvector** | Baja–Media        | Baja               | Media–Alta    | Alta                  |
| ChromaDB                  | Baja              | Baja               | Media         | Alta                  |
| Pinecone                  | Baja              | Media–Alta         | Alta          | Media                 |

## Consecuencias Positivas

- Unifica almacenamiento relacional y vectorial
- Reduce complejidad de infraestructura inicial
- Facilita trazabilidad y manejo de metadatos
- Compatible con crecimiento futuro del producto

## Consecuencias Negativas / Trade-offs

- Menor especialización que un vector DB dedicado
- Puede requerir tuning adicional a mayor escala
- La operación vectorial puede no ser tan eficiente como en soluciones altamente especializadas

## Criterios de Revisión

Esta decisión se revisará si:

- el volumen documental crece significativamente
- la latencia de retrieval supera el umbral definido
- se requiere escalado especializado para búsqueda semántica
