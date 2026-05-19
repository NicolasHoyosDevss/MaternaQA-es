# Obstetrics QA datasets

Esta carpeta contiene los datasets QA sintéticos derivados de los chunks LM de obstetricia.

## Estructura

```text
datasets/obstetrics/qa/
└── final/
    ├── dataset_summary.json      # Conteo consolidado real por split
    ├── train/
    │   ├── sft.jsonl                # Dataset final de entrenamiento en formato messages
    │   ├── raw.jsonl                # Dataset final plano para auditoría/evaluación
    │   ├── generation_report.json   # Resumen de generación
    │   ├── progress.json            # Checkpoint para reanudar generación
    │   └── progress_status.json     # Estado legible de la última ejecución
    ├── validation/
    │   └── ... mismo esquema ...
    └── test/
        └── ... mismo esquema ...
```

## Convenciones

- `sft.jsonl`: archivo que se usa para fine-tuning o entrenamiento SFT.
- `raw.jsonl`: archivo de auditoría con pregunta, respuesta, contexto fuente y metadatos.
- `generation_report.json`: métricas de generación por split.
- `progress.json` y `progress_status.json`: archivos de ejecución/reanudación; no son datos de entrenamiento.
- `final/dataset_summary.json`: resumen consolidado calculado desde los archivos finales. Úsalo para conteos finales si hubo reanudaciones.

## Artefactos experimentales

Los experimentos anteriores de comparación de modelos y diagnósticos Ragas se movieron a:

```text
artifacts/obstetrics/qa_experiments/
```

Esto mantiene aquí solo los datasets finales reales y deja los experimentos como artefactos reproducibles/auditables.
