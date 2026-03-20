# 🤖 Plantilla Oficial de Documentación — Proyecto Final AI/LLM

**Programa:** Certified AI-LLM Solution Architect  
**Curso:** 5 — Proyecto Final de Arquitectura e Integración AI/LLM  
**Documento:** Plantilla Oficial de Documentación del Proyecto (Entregable 1; Primera parte)

---

## 📋 Información General del Proyecto

| Campo                           | Valor                                                                                                     |
| ------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Nombre del Proyecto**         | PetMind AI — Personalized Pet Care Advisor API                                                            |
| **Participante(s)**             | Emilio Molina Ortiz                                                                                       |
| **Instructor**                  | Andrés Felipe Rojas Parra                                                                                 |
| **Cohorte / Edición**           | TBD                                                                                                       |
| **Fecha de Inicio**             | 06/03/2026                                                                                                |
| **Fecha de Entrega Final**      | 24/04/2026                                                                                                |
| **Versión del Documento**       | v0.1                                                                                                      |
| **Estado del Proyecto**         | En Planificación                                                                                          |
| **Repositorio GitHub/GitLab**   | https://github.com/Molinart-Designs/final-project-ai-llm                                                  |
| **Entorno Cloud**               | AWS                                                                                                       |
| **Stack Tecnológico Principal** | Python 3.11, FastAPI, OpenAI, LangChain/LlamaIndex, PostgreSQL, pgvector/ChromaDB, Docker, GitHub Actions |

---

## Tabla de Contenidos

- [1. Resumen Ejecutivo](#1-resumen-ejecutivo)
- [2. Análisis y Especificación de Requerimientos](#2-análisis-y-especificación-de-requerimientos)
- [3. Diseño de Arquitectura AI/LLM](#3-diseño-de-arquitectura-aillm)
- [4. Diseño de APIs y Conectores](#4-diseño-de-apis-y-conectores)
- [5. Seguridad, Cumplimiento y Ética](#5-seguridad-cumplimiento-y-ética)
- [6. Implementación y Configuración de Infraestructura](#6-implementación-y-configuración-de-infraestructura)
- [7. Estrategia de Pruebas y Resultados](#7-estrategia-de-pruebas-y-resultados)
- [8. Despliegue, Escalabilidad y Costos](#8-despliegue-escalabilidad-y-costos)
- [9. Observabilidad y Monitoreo](#9-observabilidad-y-monitoreo)
- [10. Resultados, Conclusiones y Trabajo Futuro](#10-resultados-conclusiones-y-trabajo-futuro)
- [11. Rúbrica de Evaluación](#11-rúbrica-de-evaluación)
- [12. Referencias y Bibliografía](#12-referencias-y-bibliografía)
- [Anexos](#anexos)

---

## 1. Resumen Ejecutivo

PetMind AI es una solución AI/LLM orientada al mercado de dueños de mascotas en México, diseñada para ofrecer orientación personalizada, contextual y accionable sobre cuidado, nutrición, comportamiento y rutinas de mascotas. El proyecto se enfoca en construir una API backend desplegada en AWS que pueda integrarse posteriormente con una aplicación móvil, pero que de manera independiente ya resuelva un caso de uso real mediante un pipeline RAG (Retrieval-Augmented Generation).

Actualmente, los dueños de mascotas suelen consultar información dispersa en buscadores, blogs, videos y redes sociales. Esto provoca respuestas genéricas, ambiguas y, en muchos casos, poco confiables. Además, la mayoría de las fuentes no toman en cuenta el contexto específico de la mascota: especie, edad, tamaño, nivel de actividad o perfil general. PetMind AI resuelve este problema permitiendo consultar en lenguaje natural y recibir respuestas basadas en conocimiento curado y adaptadas al perfil del animal.

La solución propuesta utiliza una arquitectura AI/LLM con recuperación semántica sobre documentos especializados de cuidado de mascotas. A partir de esta base, un modelo LLM genera respuestas útiles con fuentes asociadas y con restricciones explícitas para evitar diagnósticos veterinarios concluyentes o recomendaciones de alto riesgo. Esto permite una experiencia más segura, explicable y alineada con buenas prácticas de diseño de sistemas AI.

Desde el punto de vista técnico, el proyecto será implementado como una API REST con tres endpoints principales: `/api/v1/query`, `/api/v1/ingest` y `/api/v1/health`. El sistema incluirá evaluación de calidad LLM, pruebas unitarias y de integración, observabilidad, seguridad básica, despliegue en la nube y documentación formal. Aunque el producto final del negocio contempla una app móvil React Native, para fines del curso el entregable se concentra en el backend AI/LLM, ya que es el componente que mejor evidencia las capacidades de arquitectura, integración, despliegue y validación técnica exigidas por el programa.

### 1.1 Propuesta de Valor y Problema que Resuelve

PetMind AI resuelve el problema de la búsqueda fragmentada y poco personalizada de información sobre mascotas. En el escenario actual, un dueño de mascota que desea saber qué alimentación conviene a un perro senior, cómo manejar ansiedad por separación o qué rutina básica de cuidados debe seguir, normalmente consulta múltiples fuentes abiertas. Estas respuestas suelen ser amplias, poco contextualizadas y sin trazabilidad clara sobre su origen. Esto incrementa la carga cognitiva del usuario, reduce la confianza y puede derivar en decisiones incorrectas.

La propuesta de valor de PetMind AI es ofrecer un asistente inteligente especializado en mascotas que no solo responda en lenguaje natural, sino que también lo haga con base en una capa de conocimiento curado y un perfil específico de la mascota. En lugar de devolver resultados genéricos, el sistema adapta la respuesta según atributos como especie, edad, tamaño y nivel de energía. Esta capacidad de personalización convierte al sistema en una herramienta más útil que un buscador tradicional y más flexible que un sistema basado únicamente en reglas.

La estrategia AI/LLM es la adecuada porque el problema requiere comprender lenguaje natural, mantener flexibilidad frente a preguntas abiertas y sintetizar contexto de manera útil para el usuario. Un enfoque solo con reglas sería demasiado rígido, mientras que un buscador clásico no resolvería la necesidad de generar respuestas resumidas, personalizadas y explicables. Al incorporar RAG, el sistema también reduce el riesgo de alucinaciones y mejora la fidelidad al conocimiento fuente.

### 1.2 Alcance y Delimitación

| ✅ EN SCOPE                                                 | ❌ OUT OF SCOPE                        |
| ----------------------------------------------------------- | -------------------------------------- |
| API AI/LLM desplegada en AWS                                | Aplicación móvil React Native completa |
| Endpoint `/api/v1/query` para consultas en lenguaje natural | Login social completo con Google/Apple |
| Endpoint `/api/v1/ingest` para carga de documentos          | Identificación de raza por imagen      |
| Endpoint `/api/v1/health` para monitoreo                    | Diagnóstico veterinario clínico        |
| Pipeline RAG con conocimiento curado sobre mascotas         | Marketplace y pagos                    |
| Personalización por perfil básico de mascota                | Comunidad y red social de dueños       |
| Evaluación LLM, pruebas, observabilidad y CI/CD             | Soporte multi-país en v1               |
| Seguridad básica y autenticación de API                     | Fine-tuning del modelo base            |

### 1.3 Indicadores Clave de Éxito (KPIs del Proyecto)

| KPI / Métrica                        | Línea Base | Meta Objetivo  | Resultado Obtenido |
| ------------------------------------ | ---------- | -------------- | ------------------ |
| Latencia promedio (p95)              | N/A        | < 2.5 segundos | Pendiente          |
| Tasa de éxito de respuestas          | N/A        | > 90%          | Pendiente          |
| Faithfulness / Fidelidad al contexto | N/A        | > 0.85         | Pendiente          |
| Answer Relevancy                     | N/A        | > 0.80         | Pendiente          |
| Costo por 1,000 consultas (USD)      | N/A        | < USD 5        | Pendiente          |
| Cobertura de pruebas (%)             | 0%         | > 60%          | Pendiente          |
| Error rate bajo carga mínima         | N/A        | < 2%           | Pendiente          |

---

## 2. Análisis y Especificación de Requerimientos

### 2.1 Contexto del Caso de Uso Empresarial

PetMind AI se ubica en el sector de tecnología aplicada al bienestar animal y asistentes digitales personalizados. El sistema está dirigido principalmente a dueños de mascotas en México que buscan orientación confiable sobre cuidados cotidianos, nutrición, rutinas, comportamiento y atención preventiva básica. En esta primera versión, el foco principal está en perros, aunque la arquitectura se plantea para soportar otras especies a futuro.

#### 5W + H

| Elemento  | Definición                                                                                                        |
| --------- | ----------------------------------------------------------------------------------------------------------------- |
| **Who**   | Dueños de mascotas, principalmente perros, que necesitan orientación confiable y personalizada                    |
| **What**  | Un sistema AI/LLM que responde preguntas sobre cuidado de mascotas usando contexto documental y perfil del animal |
| **When**  | Durante consultas frecuentes sobre alimentación, comportamiento, rutinas, prevención y cuidados generales         |
| **Where** | Inicialmente en México, desplegado en AWS, consumido vía API                                                      |
| **Why**   | Porque la información actual está fragmentada, es genérica y rara vez considera el contexto de la mascota         |
| **How**   | Mediante una API con RAG, recuperación semántica, generación con LLM y controles de seguridad                     |

#### Flujo actual (AS-IS)

Actualmente, el usuario resuelve dudas sobre su mascota consultando buscadores, redes sociales, blogs y videos. Esta forma de trabajo es manual, dispersa y poco estructurada. Las respuestas pueden ser contradictorias, superficiales o no aplicables al caso específico del animal. Tampoco existe memoria del perfil de la mascota ni trazabilidad clara de las fuentes consultadas. El usuario hace el esfuerzo de interpretar, filtrar y decidir qué información usar.

#### Flujo propuesto (TO-BE)

El usuario envía una pregunta en lenguaje natural a PetMind AI. El sistema recibe la consulta, incorpora el contexto del perfil de la mascota, recupera documentos relevantes desde la base de conocimiento y genera una respuesta personalizada usando un modelo LLM con restricciones de seguridad y alcance. La respuesta se devuelve junto con fuentes recuperadas y metadatos básicos. Cuando el sistema detecta una consulta fuera de alcance o sensible desde el punto de vista sanitario, limita su respuesta y recomienda atención profesional.

#### Frecuencia y volumen esperado

Para la fase inicial del proyecto se asume:

- 1,000 usuarios activos mensuales como escenario de diseño del MVP
- 1 a 3 consultas por usuario por semana
- 100 a 300 documentos curados inicialmente
- 10 usuarios concurrentes como mínimo para la prueba de carga del curso

### 2.2 Requerimientos Funcionales

| ID     | Descripción del Requerimiento                                                                                                           | Prioridad | Criterio de Aceptación                                                                                                                              |
| ------ | --------------------------------------------------------------------------------------------------------------------------------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| RF-001 | El sistema debe recibir consultas en lenguaje natural y retornar respuestas personalizadas con base en contexto documental recuperado.  | Alta      | El endpoint `/api/v1/query` responde con HTTP 200, cuerpo válido y fuentes asociadas en > 90% de consultas válidas del dataset de prueba.           |
| RF-002 | El sistema debe permitir la ingesta de documentos curados relacionados con cuidado de mascotas para indexarlos en el vector store.      | Alta      | El endpoint `/api/v1/ingest` indexa correctamente al menos un lote de documentos y reporta el número de documentos procesados sin errores críticos. |
| RF-003 | El sistema debe exponer un endpoint de salud para verificar el estado del LLM API, vector store y base de datos.                        | Alta      | El endpoint `/api/v1/health` retorna estado detallado de componentes y responde en < 500 ms en condiciones normales.                                |
| RF-004 | El sistema debe aceptar contexto estructurado de la mascota, incluyendo especie, edad, tamaño y energía, para personalizar respuestas.  | Alta      | Dos consultas equivalentes con perfiles de mascota distintos retornan respuestas diferenciadas y coherentes con el contexto proporcionado.          |
| RF-005 | El sistema debe devolver las fuentes utilizadas para construir la respuesta.                                                            | Alta      | Cada respuesta generada en contexto RAG incluye al menos una fuente o chunk asociado cuando existe evidencia documental relevante.                  |
| RF-006 | El sistema debe limitar respuestas fuera de alcance, especialmente en temas de diagnóstico veterinario.                                 | Alta      | Ante preguntas médicas críticas, el sistema evita diagnósticos concluyentes y muestra advertencia o recomendación de acudir a un veterinario.       |
| RF-007 | El sistema debe registrar métricas operativas por solicitud, incluyendo latencia, tokens consumidos y estado del retrieval.             | Media     | Cada request exitoso o fallido genera logs estructurados con `trace_id`, latencia, tokens y resultado del pipeline.                                 |
| RF-008 | El sistema debe permitir filtros de contexto por categoría de conocimiento, por ejemplo nutrición, comportamiento o cuidados generales. | Media     | El endpoint `/api/v1/query` acepta filtros válidos y reduce el conjunto de documentos recuperados según la categoría solicitada.                    |

### 2.3 Requerimientos No Funcionales

| ID      | Categoría            | Descripción                                                                                     | Métrica / Umbral                                                        |
| ------- | -------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| RNF-001 | Rendimiento          | La latencia del sistema debe ser adecuada para interacción conversacional.                      | p95 < 2.5 s bajo carga normal                                           |
| RNF-002 | Escalabilidad        | El sistema debe soportar el crecimiento inicial del MVP y una carga mínima concurrente medible. | Soportar al menos 10 usuarios concurrentes en pruebas del curso         |
| RNF-003 | Seguridad            | Todos los endpoints excepto `/health` deben requerir autenticación.                             | 100% de endpoints protegidos con API Key o JWT                          |
| RNF-004 | Disponibilidad       | El sistema debe estar disponible para demostración y evaluación del instructor.                 | >= 99% en entorno de prueba desplegado                                  |
| RNF-005 | Observabilidad       | La aplicación debe registrar eventos clave del pipeline AI/LLM.                                 | Logs JSON con `timestamp`, `level`, `service`, `trace_id`, `message`    |
| RNF-006 | Calidad de respuesta | Las respuestas deben mantenerse alineadas al contexto recuperado.                               | Faithfulness > 0.85 y Answer Relevancy > 0.80                           |
| RNF-007 | Costos               | El entorno de pruebas debe permanecer en un rango controlado.                                   | < USD 80/mes en entorno de pruebas                                      |
| RNF-008 | Ética y alcance      | El sistema no debe presentarse como sustituto de atención veterinaria profesional.              | 100% de consultas críticas incluyen limitación de alcance o advertencia |

### 2.4 Restricciones y Supuestos

| Restricciones                                                                                 | Supuestos                                                                                          |
| --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| El presupuesto cloud del proyecto es limitado y debe mantenerse contenido.                    | Los usuarios tienen acceso estable a internet y a un cliente que consumirá la API.                 |
| No se deben almacenar secretos ni credenciales en el repositorio.                             | El proveedor LLM estará disponible vía API durante las pruebas y demo.                             |
| La entrega del curso se enfocará en backend AI/LLM y no en frontend móvil.                    | El corpus documental será suficiente para construir una base RAG representativa.                   |
| No se realizarán diagnósticos veterinarios clínicos ni recomendaciones concluyentes de salud. | El perfil de mascota proveído por el usuario será suficiente para personalizar respuestas básicas. |
| El sistema debe poder ejecutarse en la nube y no depender solo de localhost.                  | El volumen documental inicial será manejable con una estrategia RAG base.                          |

### 2.5 Plan de Trabajo Preliminar por Hitos

El proyecto se desarrollará en cuatro hitos alineados con la estructura oficial del curso. Cada hito produce entregables concretos y reduce el riesgo de acumulación de trabajo hacia la entrega final.

| Capítulo                                        | Sesiones | Objetivo Principal                                                                                                      | Entregables Asociados | Resultado Esperado                                             |
| ----------------------------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------- | --------------------- | -------------------------------------------------------------- |
| Capítulo 1 — Alcance y Requerimientos           | S1–S2    | Definir el problema empresarial, alcance, requerimientos funcionales y no funcionales, restricciones y stack preliminar | E1                    | Documento base del proyecto completo para iniciar arquitectura |
| Capítulo 2 — Diseño de Arquitectura             | S3–S4    | Diseñar la arquitectura AI/LLM, diagramas C4, flujo de datos, ADRs, OpenAPI inicial y estrategia RAG                    | E2                    | Blueprint técnico completo y justificado                       |
| Capítulo 3 — Implementación Funcional           | S5–S6    | Construir la API, pipeline RAG, pruebas, Docker, CI/CD y despliegue cloud                                               | E3                    | Sistema funcional desplegado y medible                         |
| Capítulo 4 — Documentación Final y Presentación | S7–S8    | Consolidar resultados, costos, observabilidad, conclusiones, video demo y versión final del repositorio                 | E4 + EV               | Entrega final completa y lista para evaluación                 |

#### Cronograma resumido

- **Semana 1–2 (E1):** definición del caso de uso, alcance, RF/RNF, restricciones, stack preliminar y estructura inicial del repositorio.
- **Semana 3–4 (E2):** diseño arquitectónico, diagramas, ADRs, especificación de APIs y decisiones de RAG.
- **Semana 5–6 (E3):** implementación del backend AI/LLM, pruebas unitarias e integración, evaluación LLM, prueba de carga y despliegue.
- **Semana 7–8 (E4 + EV):** documentación final, costos reales, observabilidad, conclusiones, roadmap y video de presentación.

#### Criterios de avance por hito

- **E1 completado** cuando el caso de uso, alcance, RF/RNF y stack preliminar estén documentados y el repositorio tenga estructura mínima.
- **E2 completado** cuando existan diagramas, ADRs, estrategia RAG y OpenAPI inicial.
- **E3 completado** cuando los endpoints `/query`, `/ingest` y `/health` funcionen en entorno cloud y existan pruebas y reportes base.
- **E4 completado** cuando toda la plantilla oficial esté llena, con resultados reales, costos, observabilidad y video demo.

### 2.6 Stack Tecnológico Preliminar y Justificación Inicial

El stack preliminar fue seleccionado buscando equilibrio entre velocidad de implementación, claridad arquitectónica, compatibilidad con el enfoque del curso y escalabilidad para evolucionar posteriormente hacia el producto completo.

| Capa                | Tecnología Seleccionada           | Justificación Inicial                                                                                                                                                                                                                                                                                         |
| ------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cloud Provider      | AWS                               | Se selecciona AWS por su madurez, amplitud de servicios administrados, escalabilidad, integración con despliegues basados en contenedores y alineación con la visión futura del producto. Además, permite crecer hacia servicios como ECS, RDS, S3, CloudWatch y SQS sin rediseñar la arquitectura principal. |
| Backend API         | FastAPI + Python 3.11             | FastAPI permite construir una API REST moderna, tipada, rápida de desarrollar y sencilla de documentar con OpenAPI. Python, además, tiene mejor alineación con el ecosistema AI/LLM del curso, frameworks de orquestación, evaluación y tooling de pruebas.                                                   |
| LLM Provider        | OpenAI                            | Se elige OpenAI por su madurez comercial, soporte robusto para generación en lenguaje natural, ecosistema estable y facilidad de integración mediante API. También permite evolucionar a flujos multimodales en futuras fases del producto.                                                                   |
| Orquestación AI/LLM | LangChain o LlamaIndex            | Estas herramientas aceleran la construcción de pipelines RAG, separación de responsabilidades, trazabilidad de prompts y experimentación controlada con retrieval, ranking y composición de contexto.                                                                                                         |
| Vector Store        | pgvector o ChromaDB               | Se evalúan ambas opciones porque ofrecen dos caminos válidos: pgvector facilita consolidar almacenamiento relacional y semántico en PostgreSQL; ChromaDB simplifica pruebas locales y prototipos RAG. La decisión final se documentará formalmente en ADR.                                                    |
| Base de Datos       | PostgreSQL                        | PostgreSQL ofrece solidez transaccional, facilidad de despliegue, compatibilidad con pgvector y capacidad de escalar hacia necesidades futuras del producto, como perfiles de usuarios, mascotas, sesiones y analítica.                                                                                       |
| Containerización    | Docker + Docker Compose           | Facilitan reproducibilidad, estandarización del entorno local y despliegue consistente entre desarrollo, prueba y producción.                                                                                                                                                                                 |
| CI/CD               | GitHub Actions                    | Permite automatizar pruebas, validaciones mínimas y despliegue desde el mismo repositorio, cumpliendo además con el requisito explícito del curso.                                                                                                                                                            |
| Observabilidad      | Logging estructurado + CloudWatch | Se considera suficiente y apropiado para la primera versión, permitiendo monitoreo básico, trazabilidad de requests y posterior evolución a dashboards y alertas más completas.                                                                                                                               |

#### Criterios usados para seleccionar el stack

1. **Compatibilidad con el curso:** se privilegió un stack que facilite cumplir con RAG, evaluación LLM, pruebas, documentación y despliegue.
2. **Escalabilidad futura:** aunque el entregable del curso se concentra en backend AI/LLM, las decisiones deben permitir evolución hacia una plataforma real para dueños de mascotas.
3. **Mantenibilidad:** se evita complejidad innecesaria en la primera versión, priorizando herramientas maduras y bien documentadas.
4. **Costo controlado:** el entorno debe poder operar en modo de pruebas con costos razonables.
5. **Tiempo de implementación:** se busca entregar valor rápido sin comprometer la claridad arquitectónica.

### 2.7 Análisis de Riesgos Preliminar

Se identifican los siguientes riesgos iniciales del proyecto para anticipar decisiones técnicas y de gestión durante el desarrollo.

| Riesgo                                                                                                                     | Impacto | Probabilidad | Mitigación Inicial                                                                                                      |
| -------------------------------------------------------------------------------------------------------------------------- | ------- | ------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Alcance excesivo al intentar construir backend AI/LLM, app móvil, visión por computadora y features de negocio en paralelo | Alto    | Alta         | Limitar el entregable del curso al backend AI/LLM con RAG y dejar app móvil e identificación visual como roadmap futuro |
| Respuestas incorrectas o alucinaciones en temas sensibles de salud animal                                                  | Alto    | Media        | Usar RAG con conocimiento curado, restricciones explícitas en prompts, guardrails y disclaimers de no diagnóstico       |
| Costos variables por consumo de LLM o mala configuración cloud                                                             | Medio   | Media        | Definir límites de tokens, top-k, entornos pequeños y monitoreo de costos desde etapas tempranas                        |
| Retrasos por subestimar tiempo de pruebas, documentación y despliegue                                                      | Alto    | Alta         | Dividir el trabajo por hitos y no posponer pruebas, documentación ni CI/CD para el final                                |
| Dependencia de corpus insuficiente o poco curado                                                                           | Medio   | Media        | Construir una base inicial pequeña pero representativa y priorizar calidad sobre cantidad                               |

---

## 3. Diseño de Arquitectura AI/LLM

### 3.1 ...

### 3.2 ...

### 3.3 ...

### 3.4 ...

## 4. Diseño de APIs y Conectores

## 5. Seguridad, Cumplimiento y Ética

## 6. Implementación y Configuración de Infraestructura

## 7. Estrategia de Pruebas y Resultados

## 8. Despliegue, Escalabilidad y Costos

## 9. Observabilidad y Monitoreo

## 10. Resultados, Conclusiones y Trabajo Futuro

## 11. Rúbrica de Evaluación

## 12. Referencias y Bibliografía
