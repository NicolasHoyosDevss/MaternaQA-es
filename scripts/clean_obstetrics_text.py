from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from obstetrics_lm_utils import (
    clean_extracted_text,
    classify_page,
    default_data_dir,
    extract_page_section,
    find_repeated_lines,
    read_jsonl,
    remove_repeated_lines,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    data_dir = default_data_dir()
    parser = argparse.ArgumentParser(description="Clean extracted obstetrics PDF pages.")
    parser.add_argument("--input", type=Path, default=data_dir / "raw_pages.jsonl")
    parser.add_argument("--output", type=Path, default=data_dir / "clean_pages.jsonl")
    parser.add_argument("--report-output", type=Path, default=data_dir / "cleaning_report.json")
    parser.add_argument("--header-threshold-ratio", type=float, default=0.28)
    parser.add_argument("--include-dropped", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def clean_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    raw_rows = read_jsonl(args.input)
    repeated = find_repeated_lines(raw_rows, threshold_ratio=args.header_threshold_ratio)
    section_by_pdf: Dict[str, str] = defaultdict(str)
    cleaned_rows: List[Dict[str, Any]] = []
    drop_counts: Counter[str] = Counter()
    kept_counts: Counter[str] = Counter()

    for row in raw_rows:
        source_pdf = str(row.get("source_pdf", ""))
        page = int(row.get("page", 0))
        text = clean_extracted_text(row.get("text", ""))
        text = remove_repeated_lines(text, repeated.get(source_pdf, []))
        text = clean_extracted_text(text)

        section = extract_page_section(text, previous_section=section_by_pdf[source_pdf])
        if section:
            section_by_pdf[source_pdf] = section

        is_kept, drop_reason, metrics = classify_page(text, page_number=page)
        if row.get("needs_ocr") is True and metrics["char_count"] < 180:
            is_kept = False
            drop_reason = "needs_ocr"

        if is_kept:
            kept_counts[source_pdf] += 1
        else:
            drop_counts[drop_reason] += 1

        cleaned = {
            "pdf_id": row.get("pdf_id"),
            "source_pdf": source_pdf,
            "source_path": row.get("source_path"),
            "page": page,
            "section": section_by_pdf[source_pdf],
            "text": text,
            "is_kept": is_kept,
            "drop_reason": drop_reason,
            "needs_ocr": bool(row.get("needs_ocr")),
            "extraction_method": row.get("extraction_method"),
            "metrics": metrics,
        }
        if is_kept or args.include_dropped:
            cleaned_rows.append(cleaned)

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "total_pages": len(raw_rows),
        "written_pages": len(cleaned_rows),
        "kept_pages": sum(kept_counts.values()),
        "dropped_pages": sum(drop_counts.values()),
        "drop_reasons": dict(sorted(drop_counts.items())),
        "kept_by_pdf": dict(sorted(kept_counts.items())),
        "repeated_lines_by_pdf": repeated,
    }
    write_json(args.report_output, report)
    return cleaned_rows


def main() -> None:
    args = parse_args()
    rows = clean_rows(args)
    count = write_jsonl(args.output, rows)
    kept = sum(1 for row in rows if row.get("is_kept") is True)
    print(f"Clean pages written: {count}")
    print(f"Kept pages: {kept}")
    print(f"Saved clean pages to: {args.output}")
    print(f"Saved cleaning report to: {args.report_output}")


if __name__ == "__main__":
    main()
