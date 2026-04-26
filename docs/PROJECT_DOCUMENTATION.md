# 🤖 Plantilla Oficial de Documentación — Proyecto Final AI/LLM

**Programa:** Certified AI-LLM Solution Architect  
**Curso:** 5 — Proyecto Final de Arquitectura e Integración AI/LLM  
**Documento:** Plantilla Oficial de Documentación del Proyecto (Entregable Final)
**Repositorio:** https://github.com/Molinart-Designs/final-project-ai-llm

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
- [Anexo A — Fallback externo de confianza](#anexo-a--fallback-externo-de-confianza)

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

### 3.1 Diagrama de Arquitectura General (Nivel C4 — Contexto y Contenedor)

> 📌 **Nota:** Diagramas C4: **contexto** en `docs/diagrams/c4-context.png` (fuente `docs/diagrams/c4-context.svg`); **contenedores** en `docs/diagrams/c4-container.png` (fuente `docs/diagrams/c4-container.svg`). Copia legada unificada: `docs/architecture/architecture_general_es.png`.  
> La arquitectura lógica del sistema se compone de una API backend AI/LLM desplegada en AWS, un pipeline RAG para recuperación semántica, un proveedor LLM externo, una base de datos relacional para persistencia operativa y un vector store para recuperación contextual.

#### Descripción general de arquitectura

PetMind AI está diseñado como un sistema backend AI/LLM desacoplado del cliente final. Aunque el producto de negocio contempla una app móvil, la arquitectura evaluable del curso se centra en una API REST que puede ser consumida por clientes externos.

A alto nivel, la solución consta de los siguientes bloques:

1. **Cliente consumidor**
   - Cliente HTTP, Postman o frontend futuro
   - Envía consultas, carga documentos y valida salud del sistema

2. **Capa API**
   - FastAPI expone endpoints `/api/v1/query`, `/api/v1/ingest` y `/api/v1/health`
   - Maneja autenticación, validación, rate limiting y logging

3. **Orquestador AI/LLM**
   - Construye el flujo de consulta
   - Ejecuta retrieval
   - Ensambla prompt con contexto y restricciones
   - Invoca el modelo LLM

4. **Pipeline RAG**
   - Ingesta documental
   - Chunking
   - Embeddings
   - Indexación y recuperación semántica

5. **Modelo LLM Base**
   - OpenAI como proveedor principal
   - Genera respuestas con base en el contexto recuperado y reglas de comportamiento

6. **Vector Store**
   - Opción principal: PostgreSQL + pgvector
   - Opción alternativa: ChromaDB
   - Almacena embeddings y chunks documentales

7. **Base de datos operativa**
   - PostgreSQL
   - Guarda sesiones, metadatos, documentos, consultas y resultados de observabilidad

8. **Observabilidad y monitoreo**
   - Logging estructurado
   - Métricas operativas
   - Health checks por componente
   - Integración posterior con CloudWatch

#### Actores externos

- Dueño de mascota / usuario final
- Administrador o curador de conocimiento
- Instructor o revisor técnico
- Proveedor externo LLM (OpenAI)

---

### 3.2 Descripción de Componentes Arquitectónicos

| Componente              | Tecnología / Servicio                       | Responsabilidad Principal                                                        | Justificación de Selección                                                            |
| ----------------------- | ------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| API Backend             | FastAPI + Python 3.11                       | Exponer endpoints REST, validación, autenticación, manejo de errores y respuesta | Alta velocidad de desarrollo, tipado, ecosistema AI fuerte y documentación automática |
| Orquestador LLM         | LangChain o LlamaIndex                      | Construir flujo RAG, composición de prompts, recuperación y llamada al LLM       | Reduce complejidad de integración y acelera experimentación                           |
| Modelo LLM Base         | OpenAI GPT-4o-mini o equivalente            | Generación de respuestas personalizadas en lenguaje natural                      | Buen balance entre costo, capacidad y velocidad                                       |
| Vector Store            | PostgreSQL + pgvector                       | Persistencia semántica y recuperación contextual                                 | Permite unificar almacenamiento estructurado y vectorial en una misma base            |
| Base de Datos Operativa | PostgreSQL                                  | Persistencia de documentos, sesiones, consultas y metadatos                      | Solución robusta, madura y escalable                                                  |
| Embeddings              | OpenAI Embeddings                           | Transformar chunks a representaciones vectoriales                                | Integración directa con el stack AI seleccionado                                      |
| Seguridad               | API Key / JWT                               | Proteger endpoints y controlar acceso                                            | Suficiente para la versión inicial del curso                                          |
| Observabilidad          | Logging JSON + métricas + CloudWatch futuro | Trazabilidad, errores, latencia y consumo de tokens                              | Permite operación y diagnóstico desde etapas tempranas                                |
| Infraestructura         | Docker + Docker Compose + AWS               | Entorno local reproducible y despliegue cloud                                    | Facilita reproducibilidad y cumplimiento del curso                                    |

---

### 3.3 Diagrama de Flujo de Datos e Integración

> 📌 **Nota:** El **flujo de datos** se entrega en `docs/diagrams/data-flow.png` (fuente: `docs/diagrams/data-flow.svg`). Copia legada: `docs/architecture/data_flow_es.png`.

#### Flujo principal de consulta

1. El cliente envía una solicitud a `POST /api/v1/query`
2. La API valida autenticación, esquema y límites básicos
3. Se extrae el contexto de la mascota y filtros opcionales
4. El retriever consulta el vector store y obtiene los chunks más relevantes
5. El orquestador construye el prompt final con:
   - system prompt
   - contexto documental
   - perfil de mascota
   - consulta del usuario
6. El sistema invoca al modelo LLM
7. Se valida el output
8. La API devuelve:
   - respuesta
   - fuentes
   - tokens usados
   - latencia
9. Se registran logs y métricas

#### Flujo de ingesta

1. El cliente envía documentos a `POST /api/v1/ingest`
2. La API valida el payload
3. El pipeline divide contenido en chunks
4. Se generan embeddings
5. Los chunks y embeddings se guardan en el vector store
6. La API devuelve el resultado de indexación

#### Flujo de salud

1. El cliente llama a `GET /api/v1/health`
2. La API verifica:
   - disponibilidad de LLM
   - conexión al vector store
   - conexión a base de datos
3. Retorna estado consolidado del sistema

---

### 3.4 Estrategia de Diseño de Prompts y RAG

#### System Prompt Base

Eres PetMind AI, un asistente especializado en cuidado de mascotas.

Tu función es responder preguntas de forma clara, útil y segura usando únicamente:

1. el contexto documental recuperado,
2. el perfil estructurado de la mascota,
3. la consulta del usuario.

RESTRICCIONES:

- No inventes información.
- Si no tienes contexto suficiente, indica claramente que no cuentas con suficiente información.
- No emitas diagnósticos veterinarios concluyentes.
- Si la consulta implica riesgo de salud, urgencia médica o señales de alarma, recomienda acudir con un veterinario.
- No presentes tus respuestas como sustituto de atención profesional.
- Prioriza respuestas prácticas, comprensibles y accionables.

FORMATO DE RESPUESTA:

- Responde en español.
- Usa tono claro, empático y profesional.
- Si aplica, estructura la respuesta en:
  1. recomendación principal
  2. puntos a observar
  3. cuándo buscar ayuda profesional

### Estrategia de Recuperación (RAG)

| Parámetro            | Valor Inicial                       | Justificación                                                |
| -------------------- | ----------------------------------- | ------------------------------------------------------------ |
| Tipo de chunking     | semántico o por longitud controlada | permite preservar sentido de recomendaciones y consejos      |
| Tamaño de chunk      | 500–800 caracteres                  | equilibrio entre contexto útil y precisión de retrieval      |
| Overlap              | 80–120 caracteres                   | reduce pérdida de contexto entre fragmentos                  |
| Modelo de embeddings | OpenAI embeddings                   | integración simple con el stack y buena calidad semántica    |
| Similitud            | cosine similarity                   | estándar en recuperación semántica                           |
| Top-k                | 4                                   | suficiente para contexto útil sin inflar demasiado el prompt |
| Umbral mínimo        | 0.75 inicial                        | evita contexto débil o irrelevante                           |
| Re-ranking           | no en v1                            | se deja para una iteración futura                            |
| Filtro por metadata  | sí                                  | categoría, especie, etapa de vida                            |

Justificación de RAG

Se selecciona una arquitectura RAG porque el sistema debe responder con base en conocimiento curado y controlado, reduciendo alucinaciones y mejorando trazabilidad. Un LLM sin retrieval produciría respuestas más genéricas y menos auditables. RAG permite actualizar el conocimiento sin reentrenar el modelo base y facilita explicar de dónde proviene la respuesta.

### Arquitectura física (equivalencias por nube)

Justificación de RAG

Se selecciona una arquitectura RAG porque el sistema debe responder con base en conocimiento curado y controlado, reduciendo alucinaciones y mejorando trazabilidad. Un LLM sin retrieval produciría respuestas más genéricas y menos auditables. RAG permite actualizar el conocimiento sin reentrenar el modelo base y facilita explicar de dónde proviene la respuesta.

### Arquitectura física (equivalencias por nube)

| Capa            | AWS             | GCP                   | Azure                         |
| --------------- | --------------- | --------------------- | ----------------------------- |
| API / Ingesta   | ECS / Lambda    | Cloud Run / Functions | Azure Functions / App Service |
| Base relacional | RDS PostgreSQL  | Cloud SQL             | Azure Database for PostgreSQL |
| Storage         | S3              | GCS                   | Blob Storage                  |
| Observabilidad  | CloudWatch      | Cloud Monitoring      | Azure Monitor                 |
| Secretos        | Secrets Manager | Secret Manager        | Key Vault                     |

## 4. Diseño de APIs y Conectores

### 4.1 Especificación de Endpoints

| Endpoint         | Método | Descripción                                                                                   | Request Body / Params                                                                      | Response Schema                                                                                         |
| ---------------- | ------ | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `/api/v1/query`  | `POST` | Recibe una consulta en lenguaje natural, ejecuta retrieval y genera respuesta contextualizada | `{"query": string, "session_id": string, "pet_profile": object, "context_filter": object}` | `{"response": string, "sources": array, "tokens_used": int, "latency_ms": float, "session_id": string}` |
| `/api/v1/ingest` | `POST` | Recibe documentos y los indexa en el vector store                                             | `{"documents": array, "source_type": string}`                                              | `{"status": string, "indexed_docs": int, "errors": array}`                                              |
| `/api/v1/health` | `GET`  | Verifica estado del sistema y sus dependencias                                                | N/A                                                                                        | `{"status": string, "components": object}`                                                              |

Se entrega especificación OpenAPI inicial en:

- `docs/api/openapi.yaml` (especificación OpenAPI única del proyecto)

### 4.2 Autenticación y Autorización (Diseño Inicial)

| Campo                      | Descripción                                                       |
| -------------------------- | ----------------------------------------------------------------- |
| **Mecanismo Auth**         | API Key en header para v1; JWT como evolución                     |
| **Proveedor de Identidad** | No aplica en v1 técnica del curso; escalable a Auth0              |
| **Gestión de Secrets**     | Variables de entorno en desarrollo; Secrets Manager en producción |
| **Rate Limiting**          | Límite básico por IP o API key                                    |
| **Roles definidos**        | `admin`, `consumer`                                               |

### 4.3 Conectores de Fuentes de Datos (Diseño Inicial)

| Fuente de Datos                | Tipo                             | Conector/SDK              | Frecuencia de Sync                | Manejo de Errores                     |
| ------------------------------ | -------------------------------- | ------------------------- | --------------------------------- | ------------------------------------- |
| Documentos curados de mascotas | Texto / markdown / PDF procesado | Pipeline local de ingesta | Bajo demanda                      | Validación + errores por documento    |
| Base de datos PostgreSQL       | SQL                              | SQLAlchemy / psycopg      | Tiempo real para datos operativos | Retry + logging                       |
| OpenAI API                     | Servicio externo LLM             | SDK oficial               | Tiempo real                       | Timeout, retry y logging estructurado |

## 5. Seguridad, Cumplimiento y Ética

### 5.1 Modelo de Amenazas y Controles de Seguridad

PetMind AI procesa consultas en lenguaje natural y genera respuestas apoyadas en un pipeline RAG. Debido a esta arquitectura, el sistema está expuesto tanto a amenazas tradicionales de APIs como a riesgos específicos de sistemas AI/LLM, especialmente prompt injection, fuga de datos vía contexto recuperado, alucinaciones y abuso del modelo. En esta primera versión se adopta un enfoque preventivo con controles de autenticación, validación de entrada, límites de uso, restricciones de prompt y filtrado básico de salida.

| Amenaza / Riesgo                               | Vector de Ataque                                                                                     | Nivel       | Control Implementado                                                                                         | Justificación Técnica                                                                                                                                    |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Prompt Injection                               | El usuario intenta forzar instrucciones como “ignora el contexto” o “revela información interna”     | **ALTO**    | Guardrails de input + system prompt restrictivo + rechazo de instrucciones fuera de alcance                  | Los sistemas RAG/LLM son sensibles a instrucciones maliciosas en lenguaje natural; el system prompt y las validaciones reducen la probabilidad de desvío |
| Data Leakage por contexto RAG                  | El sistema podría devolver fragmentos sensibles o no autorizados del corpus documental               | **ALTO**    | Corpus curado y controlado + filtrado por metadata + revisión del contenido indexado                         | El proyecto usa conocimiento controlado sobre mascotas; limitar el corpus y aplicar filtros reduce el riesgo de exposición indebida                      |
| Hallucinations / Respuestas incorrectas        | El LLM genera respuestas falsas, ambiguas o demasiado seguras sin suficiente evidencia               | **ALTO**    | Uso de RAG + umbral de similitud + respuesta conservadora cuando falta contexto + evaluación LLM             | El riesgo no se elimina, pero se reduce al condicionar la generación a contexto recuperado y medir calidad con métricas formales                         |
| Recomendaciones médicas indebidas              | El usuario formula consultas de salud animal y el sistema responde como si fuera diagnóstico clínico | **ALTO**    | Restricción explícita en prompt + disclaimers + redirección a veterinario en casos críticos                  | El sistema no debe sustituir atención profesional; este control es clave por seguridad y ética                                                           |
| Exposición de API keys o secretos              | Credenciales filtradas en código, repositorio o logs                                                 | **CRÍTICO** | Variables de entorno + `.env` excluido por `.gitignore` + `.env.example` sin valores reales                  | El repositorio público o compartido es una superficie de alto riesgo; separar secretos del código es obligatorio                                         |
| Uso abusivo del API / DoS                      | Exceso de requests, automatización abusiva o scraping de endpoints                                   | **MEDIO**   | API Key + rate limiting + logs por request                                                                   | Protege recursos limitados y ayuda a evitar consumo excesivo del proveedor LLM                                                                           |
| Ingesta de documentos inválidos o maliciosos   | Un usuario autorizado intenta cargar documentos corruptos, irrelevantes o manipulados                | **MEDIO**   | Validación de payload + control de roles para ingesta + revisión del corpus                                  | La calidad y seguridad del RAG dependen directamente del conocimiento indexado                                                                           |
| Manipulación de metadata                       | Un atacante altera campos como categoría, especie o etapa de vida para contaminar retrieval          | **MEDIO**   | Validación de esquemas y enums + saneamiento de entrada                                                      | Mantener metadatos consistentes evita recuperación errónea o abusiva                                                                                     |
| Logs con información sensible                  | Datos de entrada o contexto podrían terminar completos en logs                                       | **MEDIO**   | Logging estructurado con criterio mínimo necesario + evitar almacenar secretos y cuerpos completos sensibles | La observabilidad no debe comprometer privacidad ni seguridad operativa                                                                                  |
| Dependencia del proveedor LLM                  | Falla, timeout o degradación del servicio de OpenAI                                                  | **MEDIO**   | Health checks + manejo de errores + timeouts + respuestas degradadas controladas                             | El proveedor externo es un componente crítico; la API debe fallar de forma controlada                                                                    |
| Acceso no autorizado a endpoints protegidos    | Uso de endpoints `/query` o `/ingest` sin autenticación válida                                       | **ALTO**    | API Key en todos los endpoints excepto `/health`                                                             | Es el control mínimo exigido en esta etapa para evitar abuso y acceso no autorizado                                                                      |
| Recuperación semántica de contexto irrelevante | El retriever devuelve chunks poco relacionados que degradan la respuesta                             | **MEDIO**   | Top-k controlado + threshold de similitud + filtros por metadata                                             | Este riesgo afecta calidad más que seguridad, pero impacta directamente la confiabilidad del sistema                                                     |

#### Controles iniciales priorizados

Los controles prioritarios para la primera versión del proyecto son:

1. **Autenticación en endpoints protegidos**
   - `/query` y `/ingest` requieren API Key
   - `/health` permanece abierto solo para monitoreo básico

2. **Separación estricta de secretos**
   - uso de variables de entorno
   - archivo `.env.example` sin valores reales
   - exclusión de `.env` y credenciales en `.gitignore`

3. **Prompt seguro y restrictivo**
   - prohibición de diagnósticos concluyentes
   - prohibición de inventar información
   - instrucción explícita de responder solo con contexto suficiente

4. **Controles de retrieval**
   - corpus curado
   - filtros por metadata
   - threshold de similitud
   - top-k limitado

5. **Observabilidad mínima**
   - logs estructurados
   - trazabilidad por request
   - visibilidad de errores, latencia y consumo de tokens

#### Riesgos residuales

Aunque estos controles reducen significativamente la superficie de ataque, persisten riesgos residuales en la primera versión:

- el sistema todavía puede producir respuestas incompletas o ambiguas;
- el filtrado semántico no garantiza perfección en todos los casos;
- la protección contra prompt injection en v1 será básica y no infalible;
- la validación de salida no reemplaza una revisión humana en contextos sensibles.

Por estas razones, PetMind AI se posiciona como un sistema de **orientación inteligente**, no como una herramienta de decisión clínica ni de diagnóstico veterinario.

### 5.2 Cumplimiento Regulatorio

Aunque PetMind AI no se diseña como sistema clínico ni como expediente médico veterinario, sí procesa entradas de usuarios y genera recomendaciones que pueden influir en decisiones relacionadas con el bienestar animal. Por ello, la solución incorpora controles básicos de privacidad, seguridad y uso responsable de IA, con el objetivo de mantenerse alineada con buenas prácticas de cumplimiento y minimizar riesgos éticos y operativos.

En esta primera versión, el proyecto se enfoca en un dominio de orientación general sobre mascotas y no en diagnóstico médico. Esto reduce la carga regulatoria directa, pero no elimina la necesidad de documentar límites de uso, manejo de datos y mecanismos de mitigación.

| Regulación / Marco                            | Requerimiento Aplicable                                                                             | Control Implementado                                                                                                            | Evidencia                                               |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| Principios de privacidad de datos             | Minimizar exposición de información personal y evitar almacenamiento innecesario de datos sensibles | El sistema no requiere datos altamente sensibles para funcionar; se limita a contexto básico de mascota y consultas del usuario | Diseño funcional del sistema y esquema de request       |
| Buenas prácticas de seguridad de aplicaciones | Protección de secretos, autenticación y control de acceso                                           | Uso de variables de entorno, `.env.example`, `.gitignore`, autenticación por API Key en endpoints protegidos                    | Estructura del repositorio y configuración del proyecto |
| Uso responsable de IA                         | No presentar el sistema como autoridad clínica ni como sustituto profesional                        | Prompt restrictivo, disclaimers y límites explícitos de alcance                                                                 | Sección de prompt base y modelo de amenazas             |
| Transparencia frente al usuario               | Informar que la respuesta es generada por IA y basada en contexto documental                        | La solución se documenta como sistema AI/LLM con RAG y fuentes recuperadas                                                      | README y documentación técnica                          |
| Gestión de riesgos en recomendaciones         | Evitar decisiones automáticas de alto impacto sin revisión humana                                   | El sistema no emite diagnósticos concluyentes y deriva a veterinario en casos sensibles                                         | Prompt base, controles de salida y sección ética        |

#### Enfoque de cumplimiento adoptado

El proyecto adopta un enfoque de **cumplimiento proporcional al alcance**:

- no se procesan datos financieros ni clínicos estructurados;
- no se promete exactitud médica ni automatización de decisiones de alto impacto;
- no se entrena un modelo propio con datos privados del usuario;
- no se permite que el sistema actúe como reemplazo de un veterinario.

#### Limitaciones de cumplimiento en esta versión

- no se implementa aún un módulo formal de consentimiento explícito por categorías de datos;
- no existe aún panel de borrado o portabilidad de datos de usuario;
- no se incluye gestión avanzada de retención y auditoría completa;
- no se ha realizado una revisión legal específica por país, ya que el proyecto está en fase académica/técnica.

Aun con estas limitaciones, la arquitectura y documentación dejan una base adecuada para evolucionar hacia una solución con mayor madurez regulatoria en futuras versiones.

---

### 5.3 Marco Ético de la Solución AI

PetMind AI debe operar bajo un marco ético claro porque interactúa con usuarios que podrían depositar confianza significativa en sus respuestas. Aunque el dominio del proyecto es cuidado general de mascotas y no diagnóstico clínico, existe el riesgo de que usuarios interpreten recomendaciones como verdades absolutas o como sustituto de orientación profesional.

Por ello, el sistema se diseña bajo cuatro principios éticos centrales:

1. **Transparencia**
   - El usuario debe saber que interactúa con una solución AI/LLM.
   - Las respuestas deben estar apoyadas por fuentes recuperadas cuando sea posible.

2. **No sustitución de criterio profesional**
   - El sistema puede orientar, resumir y sugerir, pero no diagnosticar clínicamente.
   - En casos sensibles, debe recomendar atención veterinaria.

3. **Reducción de alucinaciones**
   - El sistema debe minimizar respuestas inventadas o no sustentadas.
   - Si no existe suficiente contexto, debe reconocer incertidumbre.

4. **Uso responsable del contexto del usuario**
   - El perfil de la mascota se usa para personalizar respuestas, no para inferencias invasivas ni decisiones críticas automatizadas.

| Dimensión Ética                  | Riesgo Identificado                                                                          | Mecanismo de Mitigación                                                                             |
| -------------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Transparencia                    | El usuario puede pensar que la respuesta proviene de una fuente experta humana               | Divulgación explícita de que se trata de un sistema AI/LLM y presentación de fuentes cuando existan |
| Alucinaciones                    | El modelo puede generar información falsa o demasiado segura                                 | Uso de RAG, threshold de similitud, respuesta conservadora cuando falta contexto y evaluación LLM   |
| Autoridad indebida               | El usuario podría seguir recomendaciones como si fueran diagnóstico o indicación veterinaria | Restricciones en prompt, disclaimers y redirección a atención profesional en temas sensibles        |
| Sesgo del contenido              | El corpus podría privilegiar ciertos enfoques o recomendaciones no equilibradas              | Curación manual del corpus y revisión progresiva del conocimiento cargado                           |
| Privacidad                       | Las consultas del usuario podrían incluir información innecesaria o sensible                 | Minimización de datos solicitados y control del contenido almacenado en logs                        |
| Dependencia excesiva del sistema | El usuario puede delegar decisiones importantes de cuidado sin validación externa            | Lenguaje prudente, límites de alcance y recomendación de revisión profesional cuando corresponda    |

#### Postura ética del proyecto

PetMind AI se posiciona como una herramienta de **acompañamiento inteligente**, no como un reemplazo del conocimiento veterinario ni como sistema de decisión autónoma. El objetivo es ayudar al usuario a entender mejor información relevante, no desplazar la responsabilidad humana en decisiones críticas.

#### Decisiones éticas aplicadas al diseño

- se evita lenguaje absolutista o clínico cuando no hay base suficiente;
- se prioriza utilidad y claridad sobre aparente autoridad;
- se permite que el sistema diga “no tengo suficiente información”;
- se incorpora el perfil de la mascota solo para personalizar, no para clasificar riesgo médico en forma concluyente;
- se favorece trazabilidad mediante fuentes recuperadas.

#### Evolución ética futura

En versiones futuras, sería recomendable añadir:

- mecanismos explícitos de feedback del usuario sobre respuestas inseguras o incorrectas;
- revisión humana sobre categorías de consultas sensibles;
- políticas más detalladas de retención y eliminación de datos;
- evaluación sistemática de sesgos del corpus y de calidad de respuesta por categoría;
- trazabilidad ampliada con herramientas de observabilidad LLM.

## Anexo A — Fallback externo de confianza

La arquitectura técnica alineada con el repositorio (capas L1–L3, flujo de `/api/v1/query`, feature flags, modelo `research_candidates`, seguridad, política provisional vs aprobado, refresh/TTL y modos de fallo) vive en **`docs/trusted-external-fallback.md`**.

## 6. Implementación y Configuración de Infraestructura

### 6.1 Implementación del backend

El backend de **PetMind AI — Personalized Pet Care Advisor API** fue implementado en **Python 3.11** utilizando **FastAPI** como framework principal para exponer una API REST moderna, tipada y fácil de documentar. La estructura del proyecto sigue una organización modular bajo `src/`, separando responsabilidades en las siguientes capas:

- `src/api/`: definición de endpoints, rutas y esquemas de entrada/salida.
- `src/core/`: configuración central, cliente LLM y orquestación del flujo principal.
- `src/rag/`: embeddings, retrieval, ingestion y acceso al vector store.
- `src/security/`: autenticación y guardrails de seguridad.
- `src/db/`: sesión de base de datos, modelos y bootstrap de inicialización.
- `src/research/`: componentes adicionales para fallback controlado con recuperación externa confiable.
- `src/utils/`: utilidades de logging y soporte transversal.

La API implementa los tres endpoints principales definidos en el diseño:

- `GET /api/v1/health`
- `POST /api/v1/query`
- `POST /api/v1/ingest`

Estos endpoints fueron diseñados para alinearse con el enfoque académico del proyecto, donde el entregable principal es un backend AI/LLM con arquitectura RAG y no una aplicación móvil completa.

---

### 6.2 Implementación del pipeline RAG

El pipeline RAG fue implementado con una arquitectura de recuperación y generación dividida en etapas claras:

1. **Recepción de la pregunta**
   - El usuario envía una pregunta en lenguaje natural.
   - Opcionalmente, puede incluir contexto estructurado de la mascota (`pet_profile`) y filtros (`category`, `species`, `life_stage`).

2. **Validación y autenticación**
   - FastAPI valida el payload mediante esquemas Pydantic.
   - Se requiere autenticación mediante `X-API-Key` para endpoints protegidos.

3. **Evaluación inicial de riesgo**
   - Se aplican guardrails para detectar preguntas sensibles o médicas.
   - Se determina si la respuesta debe ser más conservadora o si debe sugerirse atención veterinaria.

4. **Retrieval semántico**
   - La pregunta se transforma en embedding mediante OpenAI.
   - Ese embedding se compara contra los embeddings almacenados en PostgreSQL + pgvector.
   - Se recuperan los fragmentos más relevantes según similitud y filtros.

5. **Construcción del prompt**
   - Se genera un `system prompt` con reglas de seguridad.
   - Se construye un `user prompt` con la pregunta, perfil de la mascota y contexto recuperado.

6. **Generación de respuesta**
   - El modelo LLM genera una respuesta grounded usando únicamente el contexto recuperado.

7. **Postprocesamiento y seguridad**
   - Se ajusta la confianza (`high`, `medium`, `low`) según el score del retrieval.
   - Se agregan disclaimers.
   - Se establece `needs_vet_followup` cuando aplica.
   - Se devuelven también las `sources` utilizadas en la respuesta.

Este diseño permite minimizar alucinaciones y mantener trazabilidad entre evidencia recuperada y respuesta generada.

---

### 6.3 Base de datos y vector store

El sistema utiliza **PostgreSQL** como base de datos principal y **pgvector** como extensión para almacenamiento y búsqueda vectorial.

La tabla principal para el RAG es `document_chunks`, donde cada registro contiene:

- identificadores de documento y chunk
- texto del fragmento
- metadatos relevantes
- categoría
- especie
- etapa de vida
- embedding vectorial
- timestamps de creación

La persistencia se gestiona con **SQLAlchemy** y las migraciones con **Alembic**. Esto permite:

- trazabilidad de cambios de esquema
- inicialización reproducible
- despliegue consistente entre local y cloud

---

### 6.4 Ingesta de conocimiento

El endpoint `POST /api/v1/ingest` permite cargar conocimiento curado al sistema. El flujo de ingesta consiste en:

1. recibir uno o más documentos estructurados
2. normalizar y dividir el texto en chunks
3. generar embeddings por chunk
4. persistir contenido, metadatos y embeddings en PostgreSQL + pgvector

La estrategia de chunking usada en esta versión es por longitud controlada, con parámetros configurables:

- `CHUNK_SIZE`
- `CHUNK_OVERLAP`

Esto permite mantener consistencia entre recuperación semántica y tamaño utilizable del contexto.

---

### 6.5 Seguridad implementada

La seguridad del sistema en esta versión incluye:

- autenticación básica mediante `X-API-Key`
- separación entre endpoints públicos y protegidos
- guardrails para consultas sensibles o médicas
- restricción explícita contra diagnósticos veterinarios concluyentes
- fallback seguro cuando no existe suficiente grounding
- logging estructurado para monitoreo y depuración

Adicionalmente, el sistema fue diseñado para distinguir entre:

- respuestas con suficiente evidencia
- respuestas con contexto limitado
- respuestas que requieren derivación a veterinario

Como mejora futura, los secretos actualmente usados en despliegue serán migrados a un manejador de secretos administrado (por ejemplo, AWS Secrets Manager o Parameter Store).

---

### 6.6 Contenerización y entorno local

El proyecto fue contenedorizado con **Docker** y **Docker Compose** para facilitar reproducibilidad y despliegue.

#### Dockerfile

Se utiliza un `Dockerfile` multi-stage para:

- instalar dependencias en una etapa de build
- copiar únicamente el entorno necesario a la imagen final
- reducir tamaño y complejidad de la imagen runtime
- mantener una imagen más limpia para producción

La imagen final expone el puerto `8000` y define un `HEALTHCHECK` contra `/api/v1/health`.

#### Docker Compose

El archivo `docker-compose.yml` define:

- un servicio `postgres` basado en `pgvector/pgvector`
- un servicio `api` para FastAPI
- volúmenes persistentes
- variables de entorno desde `.env`
- health checks tanto para la base de datos como para la API
- dependencias entre servicios para arranque ordenado

Esto permite levantar localmente un entorno funcional y reproducible con una sola instrucción.

---

### 6.7 Despliegue en la nube

Para el despliegue cloud se utilizó **AWS**, siguiendo una arquitectura mínima pero defendible para el curso.

#### Servicios utilizados

- **Amazon RDS for PostgreSQL**
- **Amazon ECR**
- **Amazon ECS con Fargate**
- **Amazon CloudWatch Logs**
- **IAM**

#### Flujo de despliegue

1. La imagen Docker se construye localmente.
2. La imagen se etiqueta y se sube a **Amazon ECR**.
3. Se registra una **Task Definition** en ECS.
4. Se crea un **Service** en **ECS Fargate**.
5. El contenedor corre en AWS y expone la API públicamente.
6. La aplicación se conecta a una instancia **RDS PostgreSQL**.
7. Los logs de ejecución se envían a **CloudWatch Logs**.

El endpoint público `/api/v1/health` fue validado en el entorno desplegado y respondió correctamente con estado `healthy`.

---

### 6.8 Configuración de entornos

La configuración del sistema se centraliza en `src/core/config.py`, con soporte para variables de entorno cargadas desde `.env` en local o desde variables del entorno en cloud.

Entre las variables principales se encuentran:

- configuración de API
- proveedor LLM
- modelo de embeddings
- conexión a PostgreSQL
- parámetros de chunking
- umbral de similitud
- nivel de logging
- flags de fallback externo confiable

Esto permite separar claramente la configuración entre:

- desarrollo local
- pruebas
- staging
- producción

---

### 6.9 CI/CD y automatización

El proyecto incluye workflows de GitHub Actions para integración continua. Actualmente el pipeline ejecuta:

- instalación de dependencias
- verificación estructural del proyecto
- ejecución de pruebas
- generación de cobertura

Como parte de la evolución del proyecto, se contempla fortalecer el pipeline de CD para automatizar completamente la construcción de imagen, publicación en ECR y despliegue hacia ECS.

---

### 6.10 Estado actual de infraestructura

Al cierre de esta implementación, el sistema cuenta con:

- backend funcional bajo `src/`
- base de datos PostgreSQL con pgvector
- migraciones administradas con Alembic
- entorno local reproducible con Docker Compose
- despliegue funcional en AWS
- endpoint de salud validado en producción
- pipeline RAG operativo de extremo a extremo
- base para fallback externo confiable con feature flags

La infraestructura actual es suficiente para demostrar el funcionamiento técnico del MVP del curso y sirve también como base para una evolución posterior hacia una arquitectura más robusta.

---

## 7. Estrategia de Pruebas y Resultados

### 7.1 Enfoque general de pruebas

La estrategia de pruebas del proyecto fue diseñada para validar tanto la lógica interna del sistema como el comportamiento observable desde la API.

Se adoptó una combinación de:

- **pruebas unitarias**
- **pruebas de integración**
- **prueba de integración RAG end-to-end**
- **validaciones manuales en entorno desplegado**

El objetivo fue asegurar que el sistema no solo compila y responde, sino que también mantiene el comportamiento esperado en autenticación, guardrails, retrieval, orquestación, persistence y despliegue.

---

### 7.2 Pruebas unitarias

Las pruebas unitarias se enfocan en componentes aislados del backend. Entre las áreas cubiertas se encuentran:

- autenticación por API key
- endpoint de salud
- guardrails y clasificación de riesgo
- lógica del orquestador
- retriever
- política de promoción de conocimiento provisional

Estas pruebas validan el comportamiento interno de la aplicación sin necesidad de depender de servicios externos reales.

---

### 7.3 Pruebas de integración

Las pruebas de integración verifican el comportamiento conjunto de varias capas del sistema, incluyendo:

- endpoints de `query`
- endpoint de `ingest`
- validación de payloads
- respuestas HTTP esperadas
- integración entre rutas, esquemas y dependencias

También se añadieron pruebas de integración para la arquitectura de trusted external fallback usando mocks, con el fin de validar la lógica del flujo sin depender de internet.

---

### 7.4 Prueba de integración RAG end-to-end

Como parte de E3 se implementó una prueba de integración explícita del pipeline RAG de extremo a extremo.

Esta prueba:

- usa el endpoint real `/api/v1/query`
- utiliza autenticación real
- usa el orquestador real
- utiliza el retriever real
- consulta la base de datos real de prueba
- recupera chunks reales desde `document_chunks`
- solo controla embeddings y generación final del LLM para mantener determinismo

Con ello se validó el flujo:

**consulta → embedding → retrieval → contexto → generación → respuesta HTTP**

Esta prueba es importante porque demuestra el comportamiento real del pipeline RAG más allá de pruebas HTTP basadas solo en mocks de alto nivel.

---

### 7.5 Cobertura de pruebas

El proyecto integra **pytest-cov** para medición de cobertura.

Se generó el archivo:

- `reports/coverage.xml`

La cobertura total obtenida fue:

- **82.91%**

Este valor supera ampliamente el umbral mínimo requerido por el entregable (**60%**) y proporciona una buena señal de robustez en la base del proyecto.

La cobertura se integra también al pipeline de CI para validar automáticamente la salud de la suite.

---

### 7.6 Validación funcional del pipeline RAG

Además de las pruebas automatizadas, se realizaron validaciones funcionales manuales del flujo RAG completo.

#### Casos verificados

- ingesta de documentos curados
- almacenamiento de chunks en PostgreSQL + pgvector
- recuperación por similitud semántica
- respuestas con fuentes trazables
- respuestas con `confidence`
- respuestas con `needs_vet_followup`
- fallback cuando no existe suficiente grounding
- comportamiento conservador ante preguntas sensibles

#### Resultado

El sistema demostró ser capaz de:

- responder adecuadamente cuando existe conocimiento relevante
- no sobreafirmar cuando el contexto es insuficiente
- recomendar atención veterinaria en escenarios sensibles
- devolver trazabilidad mediante `sources`

---

### 7.7 Evaluación del comportamiento de seguridad

Los guardrails fueron probados tanto mediante tests como mediante validaciones manuales.

Se verificó que el sistema:

- no emite diagnósticos veterinarios concluyentes
- reduce confianza en preguntas sensibles
- activa `needs_vet_followup` en escenarios de riesgo
- usa fallback seguro cuando no hay contexto suficiente
- conserva trazabilidad entre evidencia recuperada y respuesta generada

Esto fue especialmente importante para mantener el carácter responsable del sistema y reducir riesgo de alucinación.

---

### 7.8 Evaluación LLM

La evaluación formal con métricas específicas de RAG/LLM se preparó como parte del entregable mediante notebook y reportes.

El objetivo de esta evaluación es medir al menos:

- groundedness / faithfulness
- answer relevance
- context relevance

El resultado esperado del proceso de evaluación será persistido en:

- `reports/ragas_report.json`

Esta evaluación complementa las pruebas tradicionales, ya que no solo verifica si el sistema responde, sino si lo hace de forma alineada con el contexto recuperado.

---

### 7.9 Prueba de carga

Como parte de la estrategia de validación no funcional, se contempla una prueba de carga con **Locust** utilizando al menos **10 usuarios concurrentes**.

El objetivo de esta prueba es observar:

- estabilidad del endpoint bajo concurrencia
- tiempos de respuesta básicos
- comportamiento general de la API en un escenario controlado

El escenario de carga está orientado principalmente a:

- `/api/v1/health`
- `/api/v1/query`

El reporte asociado se integrará como evidencia del entregable en la carpeta de reportes correspondiente.

---

### 7.10 CI y reproducibilidad

El proyecto incorpora integración continua mediante GitHub Actions, lo cual permite ejecutar de forma repetible:

- instalación de dependencias
- validaciones estructurales
- suite de pruebas
- generación de cobertura

Esto mejora la reproducibilidad del proyecto y reduce la probabilidad de regresiones silenciosas.

A nivel local, la reproducibilidad también se apoya en:

- `Dockerfile`
- `docker-compose.yml`
- `Makefile`
- migraciones Alembic
- configuración centralizada por variables de entorno

---

### 7.11 Resultados generales

A nivel técnico, el sistema alcanzó los siguientes resultados:

- backend funcional y desplegado
- endpoints operativos localmente y en la nube
- pipeline RAG demostrable
- ingesta y retrieval funcionando con pgvector
- pruebas automatizadas amplias
- cobertura superior al mínimo requerido
- integración real entre API, orquestación y base vectorial
- base sólida para evaluación LLM y pruebas de carga

En conjunto, estos resultados muestran que el proyecto pasó de una arquitectura teórica a una implementación funcional, reproducible y desplegada.

---

### 7.12 Limitaciones actuales y siguientes pasos

Aunque el sistema ya es funcional, todavía existen áreas de mejora:

- endurecimiento del manejo de secretos en AWS
- ampliación del corpus curado
- ejecución formal y persistente de RAGAS
- implementación completa del reporte de carga
- fortalecimiento del pipeline de CD
- revisión adicional de fallback externo confiable antes de activarlo en producción

Estas limitaciones no impiden la demostración del MVP, pero sí representan las líneas naturales de evolución del sistema para una versión posterior más robusta.
