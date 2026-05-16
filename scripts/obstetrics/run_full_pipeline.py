from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from utils import default_data_dir, project_root


def parse_args() -> argparse.Namespace:
    root = project_root()
    data_dir = default_data_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Run the full Spanish obstetrics LM corpus pipeline. "
            "Add PDFs to the input directory, run this script, and the final JSONL files are rebuilt."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=root / "raw_data" / "obstetrics" / "spanish")
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-clinical-score", type=int, default=5)
    parser.add_argument("--min-tokens", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--overlap-tokens", type=int, default=80)
    parser.add_argument("--samples-per-pdf", type=int, default=5)
    # Phase 7-8 optional steps
    parser.add_argument(
        "--extract-tables",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run table extraction step (requires pdfplumber).",
    )
    parser.add_argument("--table-strategy", choices=["lines", "text", "both"], default="lines")
    parser.add_argument("--table-min-rows", type=int, default=2)
    parser.add_argument("--table-min-cols", type=int, default=2)
    parser.add_argument(
        "--generate-qa",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run synthetic QA generation step (requires OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--qa-dry-run",
        action="store_true",
        help="Show QA generation cost estimate without calling API.",
    )
    return parser.parse_args()


def run_step(label: str, command: List[str]) -> None:
    print(f"\n== {label} ==", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=project_root(), check=True)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    raw_pages = data_dir / "raw_pages.jsonl"
    inventory = data_dir / "inventory.json"
    clean_pages = data_dir / "clean_pages.jsonl"
    cleaning_report = data_dir / "cleaning_report.json"
    tables = data_dir / "tables.jsonl"
    table_report = data_dir / "table_extraction_report.json"
    chunks = data_dir / "chunks.jsonl"
    train = data_dir / "train_lm.jsonl"
    validation = data_dir / "validation_lm.jsonl"
    test = data_dir / "test_lm.jsonl"
    build_report = data_dir / "build_report.json"
    audit_report = data_dir / "audit_report.json"

    run_step(
        "Extract PDFs",
        [
            sys.executable,
            "scripts/obstetrics/extract_pdfs.py",
            "--input-dir",
            str(args.input_dir),
            "--output",
            str(raw_pages),
            "--inventory-output",
            str(inventory),
            "--recursive" if args.recursive else "--no-recursive",
        ],
    )
    run_step(
        "Clean Pages",
        [
            sys.executable,
            "scripts/obstetrics/clean_text.py",
            "--input",
            str(raw_pages),
            "--output",
            str(clean_pages),
            "--report-output",
            str(cleaning_report),
            "--inventory",
            str(inventory),
        ],
    )
    if args.extract_tables:
        run_step(
            "Extract Tables",
            [
                sys.executable,
                "scripts/obstetrics/extract_tables.py",
                "--input-dir",
                str(args.input_dir),
                "--output",
                str(tables),
                "--report-output",
                str(table_report),
                "--strategy",
                str(args.table_strategy),
                "--min-rows",
                str(args.table_min_rows),
                "--min-cols",
                str(args.table_min_cols),
                "--recursive" if args.recursive else "--no-recursive",
            ],
        )
    run_step(
        "Build LM Dataset",
        [
            sys.executable,
            "scripts/obstetrics/build_lm_dataset.py",
            "--input",
            str(clean_pages),
            "--inventory",
            str(inventory),
            "--chunks-output",
            str(chunks),
            "--train-output",
            str(train),
            "--validation-output",
            str(validation),
            "--test-output",
            str(test),
            "--build-report-output",
            str(build_report),
            "--validation-ratio",
            str(args.validation_ratio),
            "--test-ratio",
            str(args.test_ratio),
            "--seed",
            str(args.seed),
            "--min-clinical-score",
            str(args.min_clinical_score),
            "--min-tokens",
            str(args.min_tokens),
            "--max-tokens",
            str(args.max_tokens),
            "--overlap-tokens",
            str(args.overlap_tokens),
        ],
    )
    run_step(
        "Audit Dataset",
        [
            sys.executable,
            "scripts/obstetrics/audit_dataset.py",
            "--raw-pages",
            str(raw_pages),
            "--clean-pages",
            str(clean_pages),
            "--chunks",
            str(chunks),
            "--train",
            str(train),
            "--validation",
            str(validation),
            "--test",
            str(test),
            "--inventory",
            str(inventory),
            "--output",
            str(audit_report),
            "--table-report",
            str(table_report),
            "--samples-per-pdf",
            str(args.samples_per_pdf),
            "--seed",
            str(args.seed),
        ],
    )

    # Phase 7: optional synthetic QA generation
    if args.generate_qa:
        qa_cmd = [
            sys.executable,
            "scripts/obstetrics/generate_synthetic_qa.py",
            "--input",
            str(train),
            "--model",
            "gpt-5.4-mini",
        ]
        if args.qa_dry_run:
            qa_cmd.append("--dry-run")
        run_step("Generate Synthetic QA", qa_cmd)

    print("\nPipeline complete.", flush=True)
    print(f"Train JSONL: {train}", flush=True)
    print(f"Validation JSONL: {validation}", flush=True)
    print(f"Test JSONL: {test}", flush=True)
    if args.extract_tables:
        print(f"Tables JSONL: {tables}", flush=True)
    print(f"Audit report: {audit_report}", flush=True)
    if args.extract_tables:
        print(f"Table report: {table_report}", flush=True)
    if args.generate_qa:
        print(f"QA output: {data_dir / 'synthetic_qa_sft.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
