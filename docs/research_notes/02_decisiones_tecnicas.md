# Decisiones técnicas registradas

| Fecha | Decisión | Motivo | Evidencia / efecto |
|---|---|---|---|
| 2026-05-16 | Priorizar SFT Q+A sobre DAPT | El objetivo real del proyecto es entrenar un modelo instructivo clínico | La discusión metodológica y el pipeline de QA apuntan a ese uso |
| 2026-05-16 | Mantener split por documento | Evita fuga de un mismo PDF entre train y validation | `leakage_detected=false` en reportes |
| 2026-05-16 | Añadir `doc_type`, `content_role`, `topics` | Permite auditar cobertura y seleccionar chunks accionables para QA | `build_report.json` y `audit_report.json` |
| 2026-05-16 | Añadir dedupe audit | Se necesitaba visibilidad explícita de duplicados inter-PDF | `dedupe_audit` en `build_report.json` |
| 2026-05-16 | Añadir `qa_id` y reporte de grounding | Hace trazable cada QA y permite filtrar respuestas poco apoyadas | `generate_synthetic_qa.py` |
| 2026-05-16 | Renombrar carpeta raw a `raw_data/obstetrics/` | `obstetrics/` contenía datos fuente, no código ni artefactos procesados | Estructura del repo más clara |

## Convenciones vigentes

- `raw_data/` = insumos fuente inmutables o casi inmutables.
- `data/` = artefactos derivados del pipeline.
- `docs/research_notes/` = decisiones, estado y evaluación del proyecto.
- No escalar volumen antes de verificar calidad, trazabilidad y evaluación.
