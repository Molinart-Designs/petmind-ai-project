# ADR-001: Usar OpenAI como modelo LLM base

**Fecha:** 05/04/2026  
**Estado:** Aceptado  
**Autores:** Emilio Molina  
**Revisado por:** _Pendiente_

## Contexto

PetMind AI requiere un modelo de lenguaje capaz de generar respuestas claras, útiles y controladas en español, a partir de contexto recuperado desde un pipeline RAG. El sistema debe operar con costos razonables, buena latencia y soporte para integración vía API. Además, se necesita una opción madura para un proyecto académico con orientación profesional, donde la facilidad de integración y la estabilidad del ecosistema sean factores importantes.

El sistema no busca entrenar un modelo desde cero ni realizar fine-tuning en esta primera versión. Por ello, la decisión del modelo base debe priorizar integración rápida, documentación sólida, buena capacidad de generación y compatibilidad con un diseño de prompts estricto.

## Decisión

Se decide usar **OpenAI** como proveedor principal de modelo LLM, con una variante optimizada para costo/rendimiento como base inicial del proyecto.

## Opciones Evaluadas

| Opción           | Latencia (aprox.) | Costo/mes (aprox.) | Escalabilidad | Facilidad integración |
| ---------------- | ----------------- | ------------------ | ------------- | --------------------- |
| **OpenAI**       | Baja–Media        | Media              | Alta          | Alta                  |
| Anthropic Claude | Baja–Media        | Media              | Alta          | Media                 |
| Gemini           | Baja–Media        | Media              | Alta          | Media                 |

## Consecuencias Positivas

- Integración rápida y bien documentada
- Buen desempeño en generación de respuestas naturales
- Ecosistema maduro para proyectos AI/LLM
- Posibilidad de evolucionar a flujos multimodales en fases futuras

## Consecuencias Negativas / Trade-offs

- Dependencia de proveedor externo
- Costo variable según uso
- Riesgo de cambios de precio o modelo con el tiempo

## Criterios de Revisión

Esta decisión se revisará si:

- el costo mensual supera el presupuesto del proyecto de forma sostenida
- la latencia p95 supera el umbral objetivo del sistema
- otro proveedor demuestra mejor relación costo/calidad para este caso
