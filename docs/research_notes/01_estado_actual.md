# Estado actual del corpus obstétrico

El proyecto construye un corpus en español para **SFT Q+A** a partir de PDFs clínicos de obstetricia y ginecología. El pipeline ya cubre extracción, limpieza, chunking, enriquecimiento semántico, partición por documento y auditoría.

## Lectura rápida

| Área | Estado actual |
|---|---|
| Objetivo principal | SFT Q+A en español |
| Datos raw | `raw_data/obstetrics/spanish/` |
| Particiones LM | `train_lm.jsonl`, `validation_lm.jsonl` |
| Riesgo de leakage por PDF | Controlado con split por documento |
| Calidad actual | Útil para piloto; aún requiere validación humana y benchmark propio |
| Próximo salto de madurez | QA sintético auditado + benchmark interno |

## Qué ya está bien encaminado

- Limpieza de páginas y descarte de ruido.
- Chunks clínicos de tamaño controlado.
- Metadatos por chunk (`pdf_id`, `chunk_id`, `doc_type`, `content_role`, `topics`).
- Split por documento para evitar fuga entre train y validation.
- Extracción de tablas disponible como artefacto paralelo.
- Trazabilidad futura de QA mediante `qa_id -> chunk_id`.

## Qué todavía debe vigilarse

- Clasificación documental: todavía puede quedar contenido en `unknown`.
- OCR: las páginas marcadas `needs_ocr` no se rescatan automáticamente.
- Grounding QA: el reporte automático es una señal inicial, no reemplaza revisión humana.
- Benchmark: no existe uno público perfectamente alineado al dominio, idioma y estilo de preguntas del proyecto.

## Interpretación metodológica

- Para **SFT Q+A**, el corpus actual puede sostener un piloto serio si los pares QA se generan y curan con cuidado.
- Para **continued pretraining/DAPT**, el volumen actual no debe presentarse como suficiente; ese no es el objetivo principal del proyecto en esta fase.
- Más PDFs ayudan solo si aumentan cobertura clínica y diversidad sin degradar calidad.
