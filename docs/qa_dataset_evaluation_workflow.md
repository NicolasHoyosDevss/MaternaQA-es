# Flujo ordenado para generar y evaluar QA sintético

## Objetivo

Construir un dataset QA clínico útil sin mezclar:

1. **filtros operativos de construcción**, y
2. **evaluación formal del artefacto final**.

## Flujo recomendado

### 1. Generar QA sintético

Se usa `scripts/generate_synthetic_qa.py`.

Este paso produce:

- `raw_*.jsonl`: artefacto auditable;
- `sft_*.jsonl`: artefacto listo para entrenamiento;
- `report_*.json`: métricas operativas del experimento.

### 2. Filtrar y auditar operativamente

El pipeline actual conserva:

- trazabilidad por `qa_id` y `chunk_id`;
- `contexto_fuente`;
- señales auxiliares del judge custom;
- controles de limpieza de preguntas/respuestas.

Estas señales ayudan a depurar, pero **no sustituyen** la evaluación formal.

### 3. Evaluar formalmente con Ragas real

Se usa `scripts/evaluate_qa_with_ragas.py` sobre los `raw_*.jsonl`.

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
  --input datasets/obstetrics/qa/raw_C_gpt52_gen_gpt55_eval.jsonl \
  --output datasets/obstetrics/qa/ragas_C_gpt52_gen_gpt55_eval.json \
  --llm-model gpt-4o-mini
```

## Lectura correcta

- **Judge custom**: filtro operativo y diagnóstico.
- **Ragas real**: evaluación formal comparable y reportable.
- **Revisión humana**: control final de defectos que las métricas no capturan bien.
