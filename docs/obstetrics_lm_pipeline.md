# Obstetrics Dataset Pipeline

Este documento describe el pipeline activo del proyecto. Los pipelines antiguos de MedQuad, BioASQ y PubMedQA fueron retirados.

## Flujo Principal

```text
PDFs en obstetrics/spanish
  -> extracción por página
  -> limpieza clínica
  -> segmentación en chunks
  -> train_lm.jsonl / validation_lm.jsonl
  -> QA sintético opcional
```

## Pipeline Incremental

Uso normal cuando agregas PDFs nuevos:

```bash
python scripts/obstetrics/run_incremental.py
```

Características:

- Detecta PDFs nuevos o modificados.
- Procesa solo esos PDFs.
- Reemplaza registros previos si un PDF cambió.
- Hace append de chunks aceptados a los JSONL existentes.
- Mantiene el manifiesto `data/obstetrics_spanish/processed_pdfs_manifest.json`.

Opciones:

```bash
python scripts/obstetrics/run_incremental.py --recursive
python scripts/obstetrics/run_incremental.py --force
python scripts/obstetrics/run_incremental.py --keep-temp
python scripts/obstetrics/run_incremental.py --input-dir path/to/pdfs
```

## Reconstrucción Completa

Úsalo cuando cambies reglas de limpieza/chunking o quieras reproducir todo desde cero:

```bash
python scripts/obstetrics/run_full_pipeline.py
```

## Pasos Internos

Los runners llaman internamente:

```bash
python scripts/obstetrics/extract_pdfs.py
python scripts/obstetrics/clean_text.py
python scripts/obstetrics/build_lm_dataset.py
python scripts/obstetrics/audit_dataset.py
```

## Salidas LM

```text
data/obstetrics_spanish/raw_pages.jsonl
data/obstetrics_spanish/clean_pages.jsonl
data/obstetrics_spanish/chunks.jsonl
data/obstetrics_spanish/train_lm.jsonl
data/obstetrics_spanish/validation_lm.jsonl
data/obstetrics_spanish/audit_report.json
```

Formato:

```json
{"text": "Texto clínico limpio...", "metadata": {"source": "obstetrics_spanish", "source_pdf": "...", "pages": [1, 2], "chunk_id": "..."}}
```

## QA Sintético

Dry run:

```bash
python scripts/obstetrics/generate_synthetic_qa.py --dry-run --limit 5
```

Generación real:

```bash
set OPENAI_API_KEY=your_key_here
python scripts/obstetrics/generate_synthetic_qa.py --limit 20
```

Salidas:

```text
data/obstetrics_spanish/synthetic_qa_raw.jsonl
data/obstetrics_spanish/synthetic_qa_sft.jsonl
data/obstetrics_spanish/.qa_generation_progress.json
```

Formato SFT:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "metadata": {"source": "obstetrics_spanish_synthetic"}}
```

El checkpoint permite reanudar. Los errores de API no se marcan como procesados, por lo que se reintentan al relanzar.

## Notas

- `PyMuPDF` es el extractor principal.
- `pdfplumber` es fallback para páginas con poco texto.
- OCR real no está incluido todavía; páginas problemáticas se marcan como `needs_ocr`.
- Revisa `audit_report.json` antes de entrenar.
