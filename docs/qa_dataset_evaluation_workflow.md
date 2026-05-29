# Flujo de Generación y Evaluación de QA Sintético

Este documento describe cómo se construye y evalúa el dataset QA de obstetricia. La regla central es no mezclar los filtros operativos usados durante la construcción con la evaluación formal del artefacto final.

## Objetivo

Construir un dataset clínico útil para fine-tuning supervisado, con pares pregunta-respuesta:

- derivados de chunks trazables;
- redactados en español clínico claro;
- respaldados por un contexto fuente;
- auditables por documento, chunk, split y métricas.

## 1. Generación de QA

Script (ejemplo para el split de entrenamiento; repetí para validation y test ajustando rutas):

```bash
python scripts/generate_synthetic_qa.py \
  --input datasets/obstetrics/lm/train_lm.jsonl \
  --raw-output datasets/obstetrics/qa/final/train/raw.jsonl \
  --sft-output datasets/obstetrics/qa/final/train/sft.jsonl \
  --progress-file datasets/obstetrics/qa/final/train/progress.json \
  --report-output datasets/obstetrics/qa/final/train/generation_report.json \
  --model gpt-5.4-mini \
  --min-pairs 2 --max-pairs 5
```

Entradas típicas:

```text
datasets/obstetrics/lm/train_lm.jsonl
datasets/obstetrics/lm/validation_lm.jsonl
datasets/obstetrics/lm/test_lm.jsonl
```

Salidas por split:

```text
datasets/obstetrics/qa/final/<split>/raw.jsonl
datasets/obstetrics/qa/final/<split>/sft.jsonl
datasets/obstetrics/qa/final/<split>/progress.json
datasets/obstetrics/qa/final/<split>/generation_report.json
```

`raw.jsonl` es el artefacto auditable. `sft.jsonl` es el artefacto conversacional listo para entrenamiento.

## 2. Reglas de Calidad Durante la Generación

El generador busca preguntas autocontenidas y evita referencias meta como:

- "según el texto";
- "según el fragmento";
- "de acuerdo con el contexto";
- "según la tabla".

También intenta evitar preguntas dobles. Cada pregunta debe tener una demanda central clara y cada respuesta debe estar respaldada por el chunk.

Tipos de pregunta esperados:

| Tipo | Propósito |
|---|---|
| `factual` | Extraer hechos clínicos concretos. |
| `definicion` | Explicar conceptos. |
| `comparacion` | Contrastar criterios, diagnósticos o manejos. |
| `razonamiento` | Conectar evidencia y decisión clínica. |
| `aplicacion` | Resolver viñetas clínicas breves. |
| `hipotetico` | Explorar escenarios condicionados por el contexto. |

## 3. Judge Operativo

El judge custom estima:

- `faithfulness`;
- `answer_relevancy`;
- `roundtrip_consistency`;
- `quality_verdict`;
- `quality_reason`.

Esta capa ayuda a depurar generación, detectar respuestas débiles y priorizar revisión manual. No reemplaza la evaluación formal.

## 4. Evaluación Formal con Ragas

Script:

```bash
python scripts/evaluate_qa_with_ragas.py \
  --input datasets/obstetrics/qa/final/train/raw.jsonl \
  --sample-size 300 \
  --seed 42 \
  --custom-judge-model gpt-5.4-mini \
  --output datasets/obstetrics/qa/final/ragas_results/final_train_sample300_eval.json
```

Métricas formales:

| Métrica | Lectura |
|---|---|
| `ragas_faithfulness` | La respuesta está respaldada por el contexto. |
| `ragas_answer_relevancy` | La respuesta atiende la pregunta planteada. |

La evaluación puede usar `--sample-size` para reducir costo y `--seed` para hacer reproducible el muestreo.

## 5. Por Qué No Usar `context_precision`

`context_precision` evalúa ranking o recuperación entre múltiples contextos candidatos. En este proyecto cada QA se genera desde un chunk fuente conocido; no se está evaluando un retriever. Por eso las métricas principales son fidelidad y relevancia.

## 6. Publicación del Dataset

Después de generar y evaluar, se preparan variantes limpias:

```bash
python scripts/prepare_qa_publication_variants.py
```

Salidas:

```text
datasets/obstetrics/qa/publication/sft_closed_book/
datasets/obstetrics/qa/publication/sft_grounded/
datasets/obstetrics/qa/publication/qa_flat_jsonl/
datasets/obstetrics/qa/publication/dataset_summary.json
```

Las variantes de publicación eliminan campos internos de evaluación cuando están vacíos o no son necesarios para entrenamiento.

## 7. Revisión Humana Recomendada

Antes de publicar o entrenar un modelo final, revisar una muestra estratificada:

- por split;
- por PDF fuente;
- por tipo documental;
- por tema clínico;
- por pares con puntajes bajos;
- por preguntas de aplicación o razonamiento.

La revisión humana debe priorizar utilidad clínica, precisión, completitud, claridad, ausencia de alucinaciones y ausencia de recomendaciones peligrosas.

## 8. Evaluación de Modelos Fine-Tuneados

La evaluación del dataset y la evaluación del modelo son capas distintas. Para comparar adapters QLoRA, primero se generan predicciones sobre el mismo `test.jsonl` y luego se evalúa la respuesta generada contra la respuesta de referencia y el contexto fuente.

Para comparar Gemma vs MedGemma:

```bash
python scripts/inference_qlora.py \
  --adapter-dir outputs/gemma4-grounded \
  --output-prefix outputs/gemma4-grounded/test

python scripts/inference_qlora.py \
  --adapter-dir outputs/medgemma-grounded \
  --output-prefix outputs/medgemma-grounded/test
```

Para evaluar las predicciones de ambos modelos:

```bash
python scripts/evaluate_model_predictions.py \
  --input outputs/gemma4-grounded/test_predictions.jsonl \
  --output outputs/gemma4-grounded/test_eval.json

python scripts/evaluate_model_predictions.py \
  --input outputs/medgemma-grounded/test_predictions.jsonl \
  --output outputs/medgemma-grounded/test_eval.json
```

Métricas recomendadas para comparar modelos:

| Métrica | Rol |
|---|---|
| `faithfulness` | La respuesta generada está respaldada por el contexto fuente. |
| `answer_relevancy` | La respuesta generada responde la pregunta. |
| `answer_correctness` | La respuesta generada coincide con la respuesta de referencia. |
| `semantic_similarity` | La respuesta generada es semánticamente similar a la referencia. |

ROUGE/BLEU pueden reportarse como métricas auxiliares de solapamiento, pero no deberían decidir el resultado principal en respuestas clínicas abiertas: penalizan paráfrasis válidas y no detectan bien alucinaciones. La comparación principal debe combinar fidelidad al contexto, corrección contra referencia, relevancia y revisión humana estratificada.

## Lectura Correcta de las Capas

| Capa | Rol |
|---|---|
| Reglas locales | Limpieza rápida y errores obvios. |
| Judge custom | Diagnóstico operativo y priorización. |
| Ragas | Evaluación formal comparable. |
| Revisión humana | Validación clínica y editorial final. |
| Evaluación de modelo | Comparación de outputs Gemma/MedGemma contra test held-out. |
