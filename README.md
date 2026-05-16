# Obstetrics PEFT Dataset Pipeline

Pipeline para construir datasets de entrenamiento a partir de PDFs de obstetricia y ginecología.

El proyecto ahora se centra en una sola fuente de conocimiento raw:

```text
raw_data/obstetrics/spanish/*.pdf
```

Y produce dos tipos de datos:

```text
1. Corpus LM: texto clínico limpio para continued training / domain adaptation.
2. QA sintético: pares pregunta-respuesta en formato chat para SFT / QLoRA.
```

## Estructura

```text
data/
  obstetrics_spanish/
    raw_pages.jsonl
    clean_pages.jsonl
    chunks.jsonl
    train_lm.jsonl
    validation_lm.jsonl
    test_lm.jsonl
    audit_report.json
    processed_pdfs_manifest.json

docs/
  obstetrics_lm_pipeline.md
  research_notes/
    README.md
    01_estado_actual.md
    02_decisiones_tecnicas.md
    03_plan_evaluacion_y_benchmark.md

raw_data/
  obstetrics/
    spanish/
      *.pdf
    english/
      *.pdf

scripts/
  obstetrics/
    run_incremental.py
    run_full_pipeline.py
    generate_synthetic_qa.py
    extract_pdfs.py
    clean_text.py
    build_lm_dataset.py
    audit_dataset.py
    utils.py
```

## Instalación

```bash
pip install -r requirements.txt
```

## Agregar Nuevos PDFs

Copia los PDFs nuevos en:

```text
raw_data/obstetrics/spanish/
```

Luego ejecuta el pipeline incremental:

```bash
python scripts/obstetrics/run_incremental.py
```

Este comando procesa solo PDFs nuevos o modificados y actualiza:

```text
data/obstetrics_spanish/chunks.jsonl
data/obstetrics_spanish/train_lm.jsonl
data/obstetrics_spanish/validation_lm.jsonl
data/obstetrics_spanish/test_lm.jsonl
```

Para buscar PDFs dentro de subcarpetas:

```bash
python scripts/obstetrics/run_incremental.py --recursive
```

Para forzar reprocesamiento incremental de todos los PDFs descubiertos:

```bash
python scripts/obstetrics/run_incremental.py --force
```

## Reconstruir Todo

Si quieres regenerar todos los artefactos desde cero:

```bash
python scripts/obstetrics/run_full_pipeline.py
```

## Generar QA Sintético

Primero estima costo y cantidad de pares sin llamar a la API:

```bash
python scripts/obstetrics/generate_synthetic_qa.py --dry-run --limit 5
```

Para generar una muestra real:

```bash
set OPENAI_API_KEY=tu_api_key
python scripts/obstetrics/generate_synthetic_qa.py --limit 20
```

Para generar sobre todo `train_lm.jsonl`:

```bash
python scripts/obstetrics/generate_synthetic_qa.py
```

Salidas:

```text
data/obstetrics_spanish/synthetic_qa_raw.jsonl
data/obstetrics_spanish/synthetic_qa_sft.jsonl
data/obstetrics_spanish/.qa_generation_progress.json
data/obstetrics_spanish/qa_generation_report.json
```

## Formatos

Corpus LM:

```json
{"text": "Texto clínico limpio...", "metadata": {"source_pdf": "...", "pages": [1, 2], "chunk_id": "..."}}
```

QA/SFT:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "metadata": {"source": "obstetrics_spanish_synthetic"}}
```

## Auditoría

Después de cada corrida revisa:

```text
data/obstetrics_spanish/audit_report.json
```

Ahí quedan conteos, páginas descartadas, páginas que requieren OCR, distribución por PDF y muestras para revisión manual.

## Documentación viva

Para entender decisiones, estado metodológico y evaluación futura, revisa:

```text
docs/research_notes/
```

Esa carpeta funciona como bitácora técnica del proyecto y debe actualizarse cuando cambien decisiones importantes del pipeline o del diseño experimental.
