# Repository Guidelines

## Project Structure & Module Organization
- `scripts/` contains the pipeline entrypoints. Use `run_full_pipeline.py` for full rebuilds, `run_incremental.py` for changed PDFs only, and `utils.py` for shared helpers.
- `pdfs/obstetrics/` stores source clinical PDFs.
- `artifacts/obstetrics/` holds intermediate outputs: `corpus/`, `metadata/`, `reports/`, and `tables/`.
- `datasets/obstetrics/` contains published outputs, mainly `lm/` and `qa/publication/`.
- `docs/` captures methodology and workflow notes; `public/` stores documentation assets.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` — create and activate the local environment.
- `pip install -r requirements.txt` — install extraction, QA, evaluation, and training dependencies.
- `python scripts/run_full_pipeline.py` — rebuild the LM pipeline from PDF extraction through audit reports.
- `python scripts/run_incremental.py --recursive` — process only new or changed PDFs.
- `python scripts/evaluate_qa_with_ragas.py --input <raw.jsonl>` — evaluate generated QA samples.
- `python scripts/train_qlora_trl.py --model-name google/gemma-4-E2B-it --dataset-variant sft_grounded --output-dir outputs/smoke --max-steps 10 --train-limit 64 --eval-limit 32` — recommended training smoke test.

## Coding Style & Naming Conventions
- Follow existing Python 3.11 style: 4-space indentation, `snake_case` for functions/files/CLI flags, and type hints on public helpers.
- Prefer `pathlib.Path`, `argparse`, and small reusable functions over inline shell logic.
- Keep comments brief and only for non-obvious pipeline or data-handling decisions.

## Testing Guidelines
- There is no dedicated `tests/` suite yet; validate changes with focused script runs and report inspection.
- For data pipeline edits, run the affected script on a small sample and review JSON outputs under `artifacts/obstetrics/reports/`.
- For training changes, use the documented smoke test before any full QLoRA run.

## Commit & Pull Request Guidelines
- Match the repo’s history with scoped Conventional Commit subjects such as `feat(pdf): ...`, `docs(readme): ...`, or `refactor(readme): ...`.
- PRs should state dataset or script impact, commands executed, required env vars, and whether generated artifacts changed.
- Include before/after counts or sample output when a PR changes corpus, QA, or evaluation behavior.

## Security & Configuration Tips
- Keep `.env` local; never commit API keys or tokens.
- Treat generated JSONL files, model outputs, and PDFs as intentional artifacts—avoid accidental bulk diffs.
