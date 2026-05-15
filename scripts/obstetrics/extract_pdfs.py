from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from utils import (
    clean_extracted_text,
    default_data_dir,
    project_root,
    slugify,
    word_count,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    root = project_root()
    data_dir = default_data_dir()
    parser = argparse.ArgumentParser(description="Extract raw text pages from Spanish obstetrics PDFs.")
    parser.add_argument("--input-dir", type=Path, default=root / "obstetrics" / "spanish")
    parser.add_argument("--output", type=Path, default=data_dir / "raw_pages.jsonl")
    parser.add_argument("--inventory-output", type=Path, default=data_dir / "inventory.json")
    parser.add_argument("--min-fallback-chars", type=int, default=120)
    parser.add_argument("--min-ocr-chars", type=int, default=80)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--pdf",
        type=Path,
        action="append",
        default=[],
        help="Specific PDF to process. Can be passed multiple times. Overrides --input-dir discovery.",
    )
    return parser.parse_args()


def require_extractors() -> Tuple[Any, Optional[Any]]:
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency: pymupdf. Install with `pip install -r requirements.txt`.") from exc

    try:
        import pdfplumber  # type: ignore
    except ImportError:
        pdfplumber = None
    return fitz, pdfplumber


def extract_with_pymupdf(page: Any) -> str:
    blocks = page.get_text("blocks")
    if blocks:
        sorted_blocks = sorted(blocks, key=lambda block: (round(block[1], 1), round(block[0], 1)))
        parts = [str(block[4]).strip() for block in sorted_blocks if len(block) >= 5 and str(block[4]).strip()]
        if parts:
            return "\n\n".join(parts)
    return str(page.get_text("text") or "")


def extract_with_pdfplumber(pdf_path: Path, page_index: int, pdfplumber: Any) -> str:
    if pdfplumber is None:
        return ""
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_index >= len(pdf.pages):
                return ""
            return pdf.pages[page_index].extract_text(layout=True) or ""
    except Exception:
        return ""


def extract_pdf(
    pdf_path: Path,
    fitz: Any,
    pdfplumber: Optional[Any],
    min_fallback_chars: int,
    min_ocr_chars: int,
    root: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    document = fitz.open(str(pdf_path))
    page_count = len(document)
    pdf_id = slugify(pdf_path.name)
    try:
        source_path = str(pdf_path.relative_to(root))
    except ValueError:
        source_path = str(pdf_path.resolve())

    inventory = {
        "pdf_id": pdf_id,
        "source_pdf": pdf_path.name,
        "source_path": source_path,
        "file_size": pdf_path.stat().st_size,
        "page_count": page_count,
        "fallback_pages": 0,
        "needs_ocr_pages": 0,
    }

    for page_index in range(page_count):
        page = document[page_index]
        raw_text = extract_with_pymupdf(page)
        cleaned_for_metrics = clean_extracted_text(raw_text)
        method = "pymupdf"

        if len(cleaned_for_metrics) < min_fallback_chars:
            fallback_text = extract_with_pdfplumber(pdf_path, page_index, pdfplumber)
            fallback_clean = clean_extracted_text(fallback_text)
            if len(fallback_clean) > len(cleaned_for_metrics):
                raw_text = fallback_text
                cleaned_for_metrics = fallback_clean
                method = "pdfplumber"
                inventory["fallback_pages"] += 1

        needs_ocr = len(cleaned_for_metrics) < min_ocr_chars
        if needs_ocr:
            inventory["needs_ocr_pages"] += 1

        rows.append(
            {
                "pdf_id": pdf_id,
                "source_pdf": pdf_path.name,
                "source_path": source_path,
                "file_size": pdf_path.stat().st_size,
                "page_count": page_count,
                "page": page_index + 1,
                "text": raw_text,
                "char_count": len(cleaned_for_metrics),
                "word_count": word_count(cleaned_for_metrics),
                "extraction_method": method,
                "needs_ocr": needs_ocr,
            }
        )
    document.close()
    return rows, inventory


def main() -> None:
    args = parse_args()
    root = project_root()
    fitz, pdfplumber = require_extractors()
    if args.pdf:
        pdfs = sorted(path.resolve() for path in args.pdf)
    else:
        pdfs = sorted(args.input_dir.rglob("*.pdf") if args.recursive else args.input_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in: {args.input_dir}")

    all_rows: List[Dict[str, Any]] = []
    inventory_rows: List[Dict[str, Any]] = []
    for pdf_path in tqdm(pdfs, desc="Extracting PDFs"):
        rows, inventory = extract_pdf(
            pdf_path=pdf_path,
            fitz=fitz,
            pdfplumber=pdfplumber,
            min_fallback_chars=args.min_fallback_chars,
            min_ocr_chars=args.min_ocr_chars,
            root=root,
        )
        all_rows.extend(rows)
        inventory_rows.append(inventory)

    count = write_jsonl(args.output, all_rows)
    write_json(
        args.inventory_output,
        {
            "pdf_count": len(inventory_rows),
            "page_count": count,
            "pdfs": inventory_rows,
        },
    )

    print(f"PDFs processed: {len(inventory_rows)}")
    print(f"Pages extracted: {count}")
    print(f"Saved raw pages to: {args.output}")
    print(f"Saved inventory to: {args.inventory_output}")


if __name__ == "__main__":
    main()
