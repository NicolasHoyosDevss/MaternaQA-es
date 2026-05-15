# Medical PEFT Dataset Pipeline

Professional preprocessing pipeline for medical QA datasets used in supervised fine-tuning (SFT) with PEFT/QLoRA workflows.

This repository currently focuses **only** on dataset engineering and preprocessing.

## Project structure

```text
medical-peft/
├── data/
│   ├── raw/
│   │   ├── medquad/
│   │   ├── bioasq/
│   │   └── pubmedqa/
│   ├── processed/
│   │   ├── medquad_chat.jsonl
│   │   ├── bioasq_chat.jsonl
│   │   └── pubmedqa_chat.jsonl
│   └── final/
│       ├── merged.jsonl
│       ├── train.jsonl
│       ├── validation.jsonl
│       └── test.jsonl
├── scripts/
│   ├── common.py
│   ├── download_datasets.py
│   ├── preprocess_medquad.py
│   ├── preprocess_bioasq.py
│   ├── preprocess_pubmedqa.py
│   ├── merge_datasets.py
│   ├── split_dataset.py
│   └── validate_dataset.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset schema

All outputs use JSONL with the following chat format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful medical AI assistant. Provide accurate and evidence-based medical information."
    },
    {
      "role": "user",
      "content": "What are the symptoms of asthma?"
    },
    {
      "role": "assistant",
      "content": "Common symptoms include wheezing, coughing, chest tightness and shortness of breath."
    }
  ],
  "metadata": {
    "source": "medquad"
  }
}
```

## Installation

```bash
cd medical-peft
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Raw data ingestion model

Preprocessing scripts use **hybrid local+HF loading**:

1. Try local files in `data/raw/<dataset>/` (JSON, JSONL, CSV, TSV).
2. If no local records are found, fallback to Hugging Face `datasets`.

You can override dataset identifiers via CLI flags:

- MedQuAD: `--hf-dataset`, `--hf-split`
- BioASQ: `--hf-dataset`, `--hf-config`, `--hf-split`
- PubMedQA: `--hf-dataset`, `--hf-config pqa_labeled`, `--hf-split`

## Processing rules implemented

- Shared fixed system prompt across all sources.
- UTF-8 JSONL output.
- Cleans control characters and excessive whitespace.
- Cleans unusual Unicode separators (`U+2028` / `U+2029`).
- Removes common MedQuAD boilerplate (HPO/Medline helper text and table headers).
- Removes duplicate/repeated assistant sentences.
- Removes common inline HTML tags and URLs in assistant text.
- Truncates assistant responses by source for token efficiency:
  - MedQuAD: max 260 words
  - BioASQ: max 320 words
  - PubMedQA: max 180 words
- Removes empty question/answer pairs.
- Deduplicates by `(user_content, assistant_content, source)`.
- Deterministic behavior with default seed `42`.

## Script usage

### 1) Download raw datasets (optional but recommended)

```bash
python scripts/download_datasets.py
```

Important args:

- `--datasets medquad bioasq pubmedqa`
- `--output-root data/raw`
- `--seed 42`
- `--force-overwrite` / `--no-force-overwrite`

You can also download only one source:

```bash
python scripts/download_datasets.py --datasets pubmedqa
```

### 2) Preprocess MedQuAD

```bash
python scripts/preprocess_medquad.py
```

Important args:

- `--input data/raw/medquad`
- `--output data/processed/medquad_chat.jsonl`
- `--seed 42`
- `--use-local-first` / `--no-use-local-first`

### 3) Preprocess BioASQ

```bash
python scripts/preprocess_bioasq.py
```

Important args:

- `--input data/raw/bioasq`
- `--output data/processed/bioasq_chat.jsonl`
- `--include-snippets` / `--no-include-snippets`
- `--max-snippets 3`
- `--max-ideal-parts 8`
- `--seed 42`

`ideal_answer` lists are deduplicated and capped by `--max-ideal-parts` before joining.
When snippets are enabled, up to 3 snippets are prepended in `Context:`.

### 4) Preprocess PubMedQA (`pqa_labeled`)

```bash
python scripts/preprocess_pubmedqa.py
```

Important args:

- `--input data/raw/pubmedqa`
- `--output data/processed/pubmedqa_chat.jsonl`
- `--hf-config pqa_labeled`
- `--seed 42`

User message format:

```text
Context:
...

Question:
...
```

### 5) Merge processed datasets

```bash
python scripts/merge_datasets.py \
  --medquad-weight 0.5 \
  --bioasq-weight 0.3 \
  --pubmedqa-weight 0.2 \
  --mode downsample_strict \
  --seed 42
```

Behavior:

- Enforces strict weighted mix using downsampling only.
- No duplication/synthetic upsampling.
- Writes `data/final/merged.jsonl`.

### 6) Split train/validation/test

```bash
python scripts/split_dataset.py \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --stratify-by-source \
  --seed 42
```

Behavior:

- Stratifies by `metadata.source` by default.
- Writes `train.jsonl`, `validation.jsonl`, `test.jsonl` in `data/final/`.

### 7) Validate outputs

```bash
python scripts/validate_dataset.py --input data/final --fail-on-error
```

Validation checks:

- `messages` exists and is non-empty list.
- message roles are valid (`system`, `user`, `assistant`).
- all message contents are non-empty strings.
- `metadata.source` exists and is non-empty.

## Recommended end-to-end run order

```bash
python scripts/preprocess_medquad.py
python scripts/preprocess_bioasq.py
python scripts/preprocess_pubmedqa.py
python scripts/merge_datasets.py
python scripts/split_dataset.py
python scripts/validate_dataset.py --input data/final --fail-on-error
```

With explicit download step first:

```bash
python scripts/download_datasets.py
python scripts/preprocess_medquad.py
python scripts/preprocess_bioasq.py
python scripts/preprocess_pubmedqa.py
python scripts/merge_datasets.py
python scripts/split_dataset.py
python scripts/validate_dataset.py --input data/final --fail-on-error
```

## Notes

- This phase intentionally excludes model training.
- If a Hugging Face dataset identifier/config changes, pass updated values via CLI arguments.
