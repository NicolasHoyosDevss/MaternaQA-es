from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from obstetrics_lm_utils import (
    accepted_for_lm,
    assign_chunk_ids,
    chunk_records,
    dedupe_chunks,
    default_data_dir,
    read_jsonl,
    split_train_validation,
    to_lm_record,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    data_dir = default_data_dir()
    parser = argparse.ArgumentParser(description="Build LM train/validation JSONL from cleaned obstetrics pages.")
    parser.add_argument("--input", type=Path, default=data_dir / "clean_pages.jsonl")
    parser.add_argument("--chunks-output", type=Path, default=data_dir / "chunks.jsonl")
    parser.add_argument("--train-output", type=Path, default=data_dir / "train_lm.jsonl")
    parser.add_argument("--validation-output", type=Path, default=data_dir / "validation_lm.jsonl")
    parser.add_argument("--build-report-output", type=Path, default=data_dir / "build_report.json")
    parser.add_argument("--min-tokens", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--overlap-tokens", type=int, default=80)
    parser.add_argument("--min-accepted-tokens", type=int, default=180)
    parser.add_argument("--min-clinical-score", type=int, default=5)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_rows = read_jsonl(args.input)
    kept_rows = [row for row in clean_rows if row.get("is_kept") is True]
    candidate_chunks = chunk_records(
        kept_rows,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )

    accepted: List[Dict[str, Any]] = []
    rejected_reasons: Counter[str] = Counter()
    for chunk in candidate_chunks:
        is_accepted, reason = accepted_for_lm(
            chunk,
            min_tokens=args.min_accepted_tokens,
            min_score=args.min_clinical_score,
        )
        if is_accepted:
            accepted.append(chunk)
        else:
            rejected_reasons[reason] += 1

    chunks = assign_chunk_ids(dedupe_chunks(accepted))
    train_chunks, validation_chunks = split_train_validation(
        chunks,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    write_jsonl(args.chunks_output, chunks)
    write_jsonl(args.train_output, [to_lm_record(chunk) for chunk in train_chunks])
    write_jsonl(args.validation_output, [to_lm_record(chunk) for chunk in validation_chunks])

    by_pdf = Counter(str(chunk.get("source_pdf", "")) for chunk in chunks)
    report = {
        "input": str(args.input),
        "chunks_output": str(args.chunks_output),
        "train_output": str(args.train_output),
        "validation_output": str(args.validation_output),
        "clean_pages_read": len(clean_rows),
        "kept_pages_read": len(kept_rows),
        "candidate_chunks": len(candidate_chunks),
        "accepted_chunks_before_dedupe": len(accepted),
        "final_chunks": len(chunks),
        "train_records": len(train_chunks),
        "validation_records": len(validation_chunks),
        "rejected_chunk_reasons": dict(sorted(rejected_reasons.items())),
        "chunks_by_pdf": dict(sorted(by_pdf.items())),
        "average_token_estimate": round(
            sum(int(chunk.get("token_estimate", 0)) for chunk in chunks) / max(1, len(chunks)),
            2,
        ),
    }
    write_json(args.build_report_output, report)

    print(f"Candidate chunks: {len(candidate_chunks)}")
    print(f"Final chunks: {len(chunks)}")
    print(f"Train records: {len(train_chunks)}")
    print(f"Validation records: {len(validation_chunks)}")
    print(f"Saved chunks to: {args.chunks_output}")
    print(f"Saved train dataset to: {args.train_output}")
    print(f"Saved validation dataset to: {args.validation_output}")
    print(f"Saved build report to: {args.build_report_output}")


if __name__ == "__main__":
    main()
