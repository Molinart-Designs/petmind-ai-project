# ADR-003: Usar FastAPI como framework principal del backend AI/LLM

**Fecha:** 05/04/2026  
**Estado:** Aceptado  
**Autores:** Emilio Molina  
**Revisado por:** _Pendiente_

## Contexto

PetMind AI requiere un framework backend que permita construir una API REST moderna, clara, mantenible y alineada con las necesidades del proyecto final del curso. La solución debe exponer endpoints como `/api/v1/query`, `/api/v1/ingest` y `/api/v1/health`, integrarse fácilmente con componentes AI/LLM, soportar validación de esquemas, documentación automática, manejo explícito de errores y pruebas automatizadas.

Además, el proyecto necesita un stack que facilite la implementación de un pipeline RAG, integración con proveedores LLM externos, observabilidad, despliegue en contenedores y evolución hacia una arquitectura más robusta en AWS. Dado que el curso evalúa no solo funcionalidad, sino también claridad arquitectónica, seguridad, documentación y reproducibilidad, la elección del framework backend afecta directamente la calidad del entregable.

También existe una restricción práctica: el proyecto debe poder desarrollarse con velocidad suficiente dentro del calendario académico, evitando complejidad innecesaria en la primera versión. Por ello, el framework seleccionado debe tener una curva de adopción razonable, buen soporte para tipado, validación y documentación, y compatibilidad con el ecosistema Python orientado a AI/LLM.

## Decisión

Se decide usar **FastAPI** como framework principal del backend AI/LLM de PetMind AI.

FastAPI será responsable de exponer la API REST, validar requests y responses, centralizar autenticación básica, manejar errores, habilitar documentación OpenAPI automática y servir como punto de integración entre la capa HTTP, el orquestador RAG, el proveedor LLM y la base de datos/vector store.

## Opciones Evaluadas

| Opción                         | Latencia (aprox.) | Costo/mes (aprox.) | Escalabilidad | Facilidad integración |
| ------------------------------ | ----------------- | ------------------ | ------------- | --------------------- |
| **FastAPI**                    | Baja              | Baja               | Alta          | Alta                  |
| Flask                          | Baja              | Baja               | Media         | Media                 |
| Django / Django REST Framework | Media             | Baja–Media         | Alta          | Media                 |

## Consecuencias Positivas

- Permite construir endpoints REST de manera rápida y con buena claridad estructural.
- Genera documentación OpenAPI/Swagger automáticamente, lo cual ayuda a cumplir con los requisitos del curso.
- Tiene integración natural con Python, lo que facilita trabajar con librerías de AI/LLM, RAG, evaluación y embeddings.
- Usa validación basada en esquemas tipados, mejorando robustez y mantenibilidad.
- Facilita pruebas, modularización y manejo explícito de errores.
- Se adapta bien a despliegues con Docker y servicios cloud como AWS ECS, Lambda o App Runner.

## Consecuencias Negativas / Trade-offs

- Requiere disciplina arquitectónica para evitar que la lógica de negocio quede mezclada con la capa HTTP.
- Tiene menos componentes “incluidos por defecto” que frameworks más pesados como Django.
- En equipos acostumbrados a Node.js o Java, puede requerir una breve curva de adopción en el ecosistema Python.
- Algunas decisiones de seguridad, estructura y observabilidad deben implementarse explícitamente y no vienen resueltas por defecto.

## Criterios de Revisión

Esta decisión se revisará si:

- el framework introduce limitaciones relevantes de rendimiento o mantenibilidad durante la implementación;
- la complejidad del proyecto crece hasta requerir una estructura más opinionada o enterprise por defecto;
- la integración con componentes AI/LLM, observabilidad o despliegue se vuelve innecesariamente costosa respecto a otra alternativa;
- el tiempo de desarrollo o mantenimiento aumenta de forma significativa por limitaciones del framework.

## Relación con la Arquitectura General

La elección de FastAPI es coherente con los objetivos del proyecto porque:

- mantiene alineación con el ecosistema Python utilizado ampliamente en AI/LLM;
- simplifica el cumplimiento de requisitos de documentación, validación y pruebas;
- reduce fricción para integrar pipelines RAG y evaluación LLM;
- permite evolucionar el backend académico hacia un servicio real consumible por la app móvil futura.
