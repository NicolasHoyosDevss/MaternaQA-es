from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import load_dataset
from tqdm import tqdm

from common import (
    build_chat_example,
    clean_assistant_for_sft,
    dedupe_by_pair,
    ensure_clean_non_empty,
    load_records_from_path,
    project_root,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Preprocess BioASQ into chat JSONL format.")
    parser.add_argument("--input", type=Path, default=root / "data" / "raw" / "bioasq")
    parser.add_argument("--output", type=Path, default=root / "data" / "processed" / "bioasq_chat.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-dataset", type=str, default="bigbio/bioasq")
    parser.add_argument("--hf-config", type=str, default=None)
    parser.add_argument("--hf-split", type=str, default="train")
    parser.add_argument("--use-local-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-snippets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-snippets", type=int, default=3)
    parser.add_argument("--max-ideal-parts", type=int, default=8)
    return parser.parse_args()


def _normalize_for_dedupe(value: str) -> str:
    value = value.lower()
    value = " ".join(value.split())
    return "".join(ch for ch in value if ch.isalnum() or ch.isspace()).strip()


def normalize_ideal_answer(value: Any, max_parts: int) -> str:
    if isinstance(value, list):
        cleaned = [ensure_clean_non_empty(v) for v in value]
        cleaned = [v for v in cleaned if v]
        deduped: List[str] = []
        seen = set()
        for part in cleaned:
            norm = _normalize_for_dedupe(part)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            deduped.append(part)
            if len(deduped) >= max_parts:
                break
        return "\n\n".join(deduped)
    return ensure_clean_non_empty(value)


def extract_snippets(record: Dict[str, Any], max_snippets: int) -> List[str]:
    raw = record.get("snippets")
    if not isinstance(raw, list):
        return []

    snippets: List[str] = []
    for item in raw:
        if isinstance(item, dict):
            candidate = item.get("text") or item.get("snippet") or item.get("body")
        else:
            candidate = item
        cleaned = ensure_clean_non_empty(candidate)
        if cleaned:
            snippets.append(cleaned)
        if len(snippets) >= max_snippets:
            break
    return snippets


def build_user_message(body: str, snippets: List[str]) -> str:
    if not snippets:
        return body
    context = "\n".join(f"- {snippet}" for snippet in snippets)
    return f"Context:\n{context}\n\nQuestion:\n{body}"


def extract_fields(
    record: Dict[str, Any],
    include_snippets: bool,
    max_snippets: int,
    max_ideal_parts: int,
) -> tuple[str, str]:
    body = ensure_clean_non_empty(record.get("body") or record.get("question"))
    answer = normalize_ideal_answer(record.get("ideal_answer") or record.get("answer"), max_parts=max_ideal_parts)

    snippets = extract_snippets(record, max_snippets=max_snippets) if include_snippets else []
    user_message = build_user_message(body, snippets)
    return user_message, answer


def load_local_records(input_path: Path) -> List[Dict[str, Any]]:
    if not input_path.exists():
        return []
    return load_records_from_path(input_path)


def load_hf_records(dataset_name: str, split: str, config: str | None) -> Iterable[Dict[str, Any]]:
    if config:
        ds = load_dataset(dataset_name, config, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    for row in ds:
        yield dict(row)


def main() -> None:
    args = parse_args()

    raw_records: List[Dict[str, Any]] = []
    if args.use_local_first:
        raw_records = load_local_records(args.input)

    if not raw_records:
        raw_records = list(load_hf_records(args.hf_dataset, args.hf_split, args.hf_config))

    processed = []
    dropped = 0

    for record in tqdm(raw_records, desc="Processing BioASQ"):
        user_message, answer = extract_fields(
            record,
            include_snippets=args.include_snippets,
            max_snippets=args.max_snippets,
            max_ideal_parts=args.max_ideal_parts,
        )

        user_message = ensure_clean_non_empty(user_message)
        answer = clean_assistant_for_sft(answer, user_text=user_message, source="bioasq")

        if not user_message or not answer:
            dropped += 1
            continue

        processed.append(build_chat_example(user_message, answer, source="bioasq"))

    deduped = dedupe_by_pair(processed)
    write_jsonl(args.output, deduped)

    print(f"BioASQ raw records: {len(raw_records)}")
    print(f"BioASQ dropped invalid: {dropped}")
    print(f"BioASQ after dedupe: {len(deduped)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
