# Flujo ordenado para generar y evaluar QA sintético

## Objetivo

Construir un dataset QA clínico útil sin mezclar:

1. **filtros operativos de construcción**, y
2. **evaluación formal del artefacto final**.

## Flujo recomendado

### 1. Generar QA sintético

Se usa `scripts/generate_synthetic_qa.py`.

Este paso produce:

- `datasets/obstetrics/qa/final/<split>/raw.jsonl`: artefacto auditable;
- `datasets/obstetrics/qa/final/<split>/sft.jsonl`: artefacto listo para entrenamiento;
- `datasets/obstetrics/qa/final/<split>/generation_report.json`: métricas operativas del split.

### 2. Filtrar y auditar operativamente

El pipeline actual conserva:

- trazabilidad por `qa_id` y `chunk_id`;
- `contexto_fuente`;
- señales auxiliares del judge custom;
- controles de limpieza de preguntas/respuestas.

Estas señales ayudan a depurar, pero **no sustituyen** la evaluación formal.

### 3. Evaluar formalmente con Ragas real

Se usa `scripts/evaluate_qa_with_ragas.py` sobre los `raw.jsonl` de cada split.

Métricas formales que se reportan:

- `ragas_faithfulness`;
- `ragas_answer_relevancy`.

No se usa `context_precision` porque aquí no se evalúa retrieval/ranking de múltiples contextos.

### 4. Revisar manualmente una muestra

Antes de escalar o entregar:

- revisar una muestra estratificada;
- priorizar pares con scores bajos;
- revisar limpieza estilística, completitud, utilidad clínica y preguntas dobles.

## Comando mínimo

```bash
python scripts/evaluate_qa_with_ragas.py \
  --input datasets/obstetrics/qa/final/train/raw.jsonl \
  --sample-size 300 \
  --seed 42 \
  --custom-judge-model gpt-5.4-mini \
  --output datasets/obstetrics/qa/final/train/sample300_eval.json
```

## Lectura correcta

- **Judge custom**: filtro operativo y diagnóstico.
- **Ragas real**: evaluación formal comparable y reportable.
- **Revisión humana**: control final de defectos que las métricas no capturan bien.
