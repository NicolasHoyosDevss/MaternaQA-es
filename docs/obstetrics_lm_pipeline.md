# Pipeline LM de Obstetricia

Este documento describe el pipeline activo para convertir PDFs clínicos de obstetricia y ginecología en un corpus limpio, auditable y segmentado para entrenamiento o generación de QA.

Los pipelines antiguos de MedQuad, BioASQ y PubMedQA no forman parte del flujo actual.

## Flujo Principal

```text
pdfs/obstetrics/*.pdf
  -> extract_pdfs.py
  -> clean_text.py
  -> build_lm_dataset.py
  -> audit_dataset.py
  -> datasets/obstetrics/lm/{train,validation,test}_lm.jsonl
```

Opcionalmente, el pipeline también extrae tablas y genera QA sintético.

## 1. Extracción

Script:

```bash
python scripts/extract_pdfs.py
```

Responsabilidades:

- descubrir PDFs en `pdfs/obstetrics/`;
- extraer texto por página con `PyMuPDF`;
- usar `pdfplumber` como fallback si la extracción principal produce poco texto;
- marcar páginas con posible necesidad de OCR;
- crear o actualizar el manifiesto documental.

Salidas:

```text
artifacts/obstetrics/corpus/raw_pages.jsonl
artifacts/obstetrics/metadata/inventory.json
```

## 2. Limpieza

Script:

```bash
python scripts/clean_text.py
```

Responsabilidades:

- remover encabezados y pies repetidos;
- normalizar ruido de extracción;
- detectar secciones;
- filtrar páginas no clínicas o no útiles;
- conservar razón de descarte para auditoría.

Salidas:

```text
artifacts/obstetrics/corpus/clean_pages.jsonl
artifacts/obstetrics/reports/cleaning_report.json
```

Razones de descarte esperadas:

- `too_short`;
- `needs_ocr`;
- `fragmented_text`;
- `reference_heavy`;
- `index_or_table_of_contents`;
- `non_clinical_section`.

## 3. Chunking y Splits

Script:

```bash
python scripts/build_lm_dataset.py
```

Responsabilidades:

- agrupar páginas limpias en chunks;
- aplicar límites de tokens y solapamiento;
- deduplicar contenido exacto o casi duplicado;
- enriquecer chunks con tipo de sección, rol de contenido y temas;
- filtrar por puntaje clínico mínimo;
- crear splits a nivel de documento para evitar fuga entre train, validation y test.

Salidas:

```text
artifacts/obstetrics/corpus/chunks.jsonl
datasets/obstetrics/lm/train_lm.jsonl
datasets/obstetrics/lm/validation_lm.jsonl
datasets/obstetrics/lm/test_lm.jsonl
artifacts/obstetrics/reports/build_report.json
```

Formato LM:

```json
{
  "text": "Texto clínico limpio...",
  "metadata": {
    "source": "obstetrics_spanish",
    "source_pdf": "documento.pdf",
    "pages": [1, 2],
    "chunk_id": "documento_00001"
  }
}
```

## 4. Auditoría

Script:

```bash
python scripts/audit_dataset.py
```

Responsabilidades:

- consolidar conteos de páginas, chunks y splits;
- auditar fuga entre splits por PDF;
- resumir cobertura temática;
- resumir distribución por tipo documental;
- reportar documentos excluidos;
- incluir resumen de extracción de tablas.

Salida:

```text
artifacts/obstetrics/reports/audit_report.json
```

El reporte debe revisarse antes de entrenar o publicar un dataset.

## Pipeline Incremental

Uso normal al agregar PDFs:

```bash
python scripts/run_incremental.py
```

Características:

- detecta PDFs nuevos o modificados;
- procesa solo esos archivos;
- reemplaza registros previos de PDFs modificados;
- conserva chunks existentes no afectados;
- actualiza `processed_pdfs_manifest.json`.

Opciones útiles:

```bash
python scripts/run_incremental.py --recursive
python scripts/run_incremental.py --force
python scripts/run_incremental.py --keep-temp
python scripts/run_incremental.py --input-dir path/to/pdfs
```

## Reconstrucción Completa

Usar cuando cambien reglas de limpieza, chunking, scoring o split:

```bash
python scripts/run_full_pipeline.py
```

Opciones frecuentes:

```bash
python scripts/run_full_pipeline.py --recursive
python scripts/run_full_pipeline.py --no-extract-tables
python scripts/run_full_pipeline.py --generate-qa
python scripts/run_full_pipeline.py --qa-dry-run --generate-qa
```

## Extracción de Tablas

Script:

```bash
python scripts/extract_tables.py
```

Salidas:

```text
artifacts/obstetrics/tables/tables.jsonl
artifacts/obstetrics/reports/table_extraction_report.json
```

Las tablas se mantienen como artefacto auditable separado del texto principal. No todas las tablas son adecuadas para QA sin revisión, por lo que conviene tratarlas como evidencia complementaria.

## Buenas Prácticas

- Agregar PDFs en `pdfs/obstetrics/` y procesarlos con el runner incremental.
- Revisar `audit_report.json` después de cada corrida importante.
- Usar reconstrucción completa si cambian reglas de limpieza o chunking.
- Mantener splits por documento para evitar contaminación entre entrenamiento y evaluación.
- Documentar decisiones metodológicas en `docs/research_notes/`.
