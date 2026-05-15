from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from utils import default_data_dir, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    data_dir = default_data_dir()
    parser = argparse.ArgumentParser(description="Audit the Spanish obstetrics LM dataset artifacts.")
    parser.add_argument("--raw-pages", type=Path, default=data_dir / "raw_pages.jsonl")
    parser.add_argument("--clean-pages", type=Path, default=data_dir / "clean_pages.jsonl")
    parser.add_argument("--chunks", type=Path, default=data_dir / "chunks.jsonl")
    parser.add_argument("--train", type=Path, default=data_dir / "train_lm.jsonl")
    parser.add_argument("--validation", type=Path, default=data_dir / "validation_lm.jsonl")
    parser.add_argument("--output", type=Path, default=data_dir / "audit_report.json")
    parser.add_argument("--samples-per-pdf", type=int, default=5)
    parser.add_argument("--sample-chars", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_if_exists(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def preview(text: str, max_chars: int) -> str:
    value = " ".join(str(text).split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def validate_lm_records(rows: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    for idx, row in enumerate(rows, start=1):
        if "messages" in row:
            errors.append(f"line {idx}: unexpected messages field")
        if not isinstance(row.get("text"), str) or not row.get("text", "").strip():
            errors.append(f"line {idx}: empty text")
        if not isinstance(row.get("metadata"), dict):
            errors.append(f"line {idx}: missing metadata")
    return errors[:50]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    raw_pages = load_if_exists(args.raw_pages)
    clean_pages = load_if_exists(args.clean_pages)
    chunks = load_if_exists(args.chunks)
    train = load_if_exists(args.train)
    validation = load_if_exists(args.validation)

    raw_by_pdf = Counter(str(row.get("source_pdf", "")) for row in raw_pages)
    kept_pages = [row for row in clean_pages if row.get("is_kept") is True]
    dropped_pages = [row for row in clean_pages if row.get("is_kept") is False]
    dropped_by_reason = Counter(str(row.get("drop_reason", "")) for row in dropped_pages)
    needs_ocr = [row for row in raw_pages if row.get("needs_ocr") is True]
    chunks_by_pdf = Counter(str(row.get("source_pdf", "")) for row in chunks)

    samples_by_pdf: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    chunks_grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_grouped[str(chunk.get("source_pdf", ""))].append(chunk)

    for source_pdf, pdf_chunks in sorted(chunks_grouped.items()):
        sample_size = min(args.samples_per_pdf, len(pdf_chunks))
        for chunk in rng.sample(pdf_chunks, sample_size):
            samples_by_pdf[source_pdf].append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "pages": chunk.get("pages"),
                    "section": chunk.get("section"),
                    "token_estimate": chunk.get("token_estimate"),
                    "clinical_score": chunk.get("clinical_score"),
                    "text_preview": preview(str(chunk.get("text", "")), args.sample_chars),
                }
            )

    accepted_examples = [
        {
            "chunk_id": chunk.get("chunk_id"),
            "source_pdf": chunk.get("source_pdf"),
            "pages": chunk.get("pages"),
            "text_preview": preview(str(chunk.get("text", "")), args.sample_chars),
        }
        for chunk in chunks[:10]
    ]
    discarded_examples = [
        {
            "source_pdf": row.get("source_pdf"),
            "page": row.get("page"),
            "drop_reason": row.get("drop_reason"),
            "text_preview": preview(str(row.get("text", "")), args.sample_chars),
        }
        for row in dropped_pages[:10]
    ]

    report = {
        "raw_pages_path": str(args.raw_pages),
        "clean_pages_path": str(args.clean_pages),
        "chunks_path": str(args.chunks),
        "train_path": str(args.train),
        "validation_path": str(args.validation),
        "pdfs_processed": len(raw_by_pdf),
        "pages_total": len(raw_pages),
        "pages_kept": len(kept_pages),
        "pages_discarded": len(dropped_pages),
        "pages_needing_ocr": len(needs_ocr),
        "drop_reasons": dict(sorted(dropped_by_reason.items())),
        "chunks_generated": len(chunks),
        "train_records": len(train),
        "validation_records": len(validation),
        "distribution_by_pdf": dict(sorted(chunks_by_pdf.items())),
        "average_chunk_tokens": round(
            sum(int(chunk.get("token_estimate", 0)) for chunk in chunks) / max(1, len(chunks)),
            2,
        ),
        "lm_validation_errors": {
            "train": validate_lm_records(train),
            "validation": validate_lm_records(validation),
        },
        "accepted_examples": accepted_examples,
        "discarded_page_examples": discarded_examples,
        "manual_review_samples_by_pdf": samples_by_pdf,
    }
    write_json(args.output, report)

    print(f"PDFs processed: {report['pdfs_processed']}")
    print(f"Pages total: {report['pages_total']}")
    print(f"Pages kept: {report['pages_kept']}")
    print(f"Pages discarded: {report['pages_discarded']}")
    print(f"Pages needing OCR: {report['pages_needing_ocr']}")
    print(f"Chunks generated: {report['chunks_generated']}")
    print(f"Train records: {report['train_records']}")
    print(f"Validation records: {report['validation_records']}")
    print(f"Saved audit report to: {args.output}")


if __name__ == "__main__":
    main()
