# Spanish Obstetrics LM Corpus Pipeline

This pipeline builds a continued-training/domain-adaptation corpus from the PDFs in `obstetrics/spanish`.
It is independent from the QA/chat preprocessing scripts and produces plain language-modeling JSONL records.

## Install

```bash
pip install -r requirements.txt
```

## Run

Recommended one-command pipeline:

```bash
python scripts/run_obstetrics_lm_pipeline.py
```

To add new documentation, copy the new `.pdf` files into `obstetrics/spanish` and run the same command again.
The pipeline rebuilds the artifacts from all PDFs in that folder, so the final JSONL files include the new documents without blindly appending duplicates.

For faster iteration, use the incremental pipeline instead:

```bash
python scripts/run_obstetrics_lm_incremental.py
```

The incremental pipeline detects PDFs that are new or changed since the previous run, processes only those files, and appends their accepted chunks to the existing `chunks.jsonl`, `train_lm.jsonl`, and `validation_lm.jsonl`. It stores the processed file fingerprints in:

```text
data/obstetrics_spanish/processed_pdfs_manifest.json
```

Use this when you are adding a few new PDFs and want to avoid reprocessing the whole folder.

If you intentionally replaced an existing PDF and want to refresh only that document, the incremental pipeline detects the changed file size or modified timestamp and replaces that PDF's previous records before appending the new ones.

To force refresh all discovered PDFs through the incremental path:

```bash
python scripts/run_obstetrics_lm_incremental.py --force
```

If PDFs are organized in subfolders:

```bash
python scripts/run_obstetrics_lm_pipeline.py --recursive
python scripts/run_obstetrics_lm_incremental.py --recursive
```

You can also process a different input folder:

```bash
python scripts/run_obstetrics_lm_pipeline.py --input-dir path/to/new_pdfs
```

Manual step-by-step execution:

```bash
python scripts/extract_obstetrics_pdfs.py
python scripts/clean_obstetrics_text.py
python scripts/build_obstetrics_lm_dataset.py
python scripts/audit_obstetrics_dataset.py
```

## Outputs

```text
data/obstetrics_spanish/raw_pages.jsonl
data/obstetrics_spanish/inventory.json
data/obstetrics_spanish/clean_pages.jsonl
data/obstetrics_spanish/cleaning_report.json
data/obstetrics_spanish/chunks.jsonl
data/obstetrics_spanish/train_lm.jsonl
data/obstetrics_spanish/validation_lm.jsonl
data/obstetrics_spanish/build_report.json
data/obstetrics_spanish/audit_report.json
```

Final LM records use this shape:

```json
{"text": "Texto clinico limpio...", "metadata": {"source": "obstetrics_spanish", "source_pdf": "...", "pages": [1, 2], "section": "...", "chunk_id": "..."}}
```

The final dataset does not contain `messages`, `system`, `user`, or `assistant` fields and does not generate QA examples.

## Notes

- `PyMuPDF` is the primary extractor.
- `pdfplumber` is used as a fallback when a page has little extracted text.
- OCR is not run in v1. Pages that still have very little text are marked with `needs_ocr`.
- Review `audit_report.json` before training, especially `manual_review_samples_by_pdf`.
