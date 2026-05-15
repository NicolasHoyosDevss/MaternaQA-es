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
    parser = argparse.ArgumentParser(description="Preprocess PubMedQA (pqa_labeled) into chat JSONL format.")
    parser.add_argument("--input", type=Path, default=root / "data" / "raw" / "pubmedqa")
    parser.add_argument("--output", type=Path, default=root / "data" / "processed" / "pubmedqa_chat.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-dataset", type=str, default="qiaojin/PubMedQA")
    parser.add_argument("--hf-config", type=str, default="pqa_labeled")
    parser.add_argument("--hf-split", type=str, default="train")
    parser.add_argument("--use-local-first", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def normalize_context(value: Any) -> str:
    if isinstance(value, list):
        parts = [ensure_clean_non_empty(v) for v in value]
        parts = [v for v in parts if v]
        return "\n".join(parts)
    if isinstance(value, dict):
        candidates = []
        for key in ("contexts", "context", "abstract", "text"):
            if key in value:
                normalized = normalize_context(value[key])
                if normalized:
                    candidates.append(normalized)
        return "\n".join(candidates)
    return ensure_clean_non_empty(value)


def extract_fields(record: Dict[str, Any]) -> tuple[str, str, str]:
    question = ensure_clean_non_empty(record.get("question"))
    answer = ensure_clean_non_empty(record.get("long_answer") or record.get("answer"))

    raw_context = record.get("context")
    if raw_context is None:
        raw_context = record.get("contexts")
    context = normalize_context(raw_context)

    user_message = f"Context:\n{context}\n\nQuestion:\n{question}"
    return question, user_message, answer


def load_local_records(input_path: Path) -> List[Dict[str, Any]]:
    if not input_path.exists():
        return []
    return load_records_from_path(input_path)


def load_hf_records(dataset_name: str, config: str | None, split: str) -> Iterable[Dict[str, Any]]:
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
        raw_records = list(load_hf_records(args.hf_dataset, args.hf_config, args.hf_split))

    processed = []
    dropped = 0

    for record in tqdm(raw_records, desc="Processing PubMedQA"):
        question, user_message, answer = extract_fields(record)
        user_message = ensure_clean_non_empty(user_message)
        answer = clean_assistant_for_sft(answer, user_text=question, source="pubmedqa")

        if not question or not answer or not user_message:
            dropped += 1
            continue

        processed.append(build_chat_example(user_message, answer, source="pubmedqa"))

    deduped = dedupe_by_pair(processed)
    write_jsonl(args.output, deduped)

    print(f"PubMedQA raw records: {len(raw_records)}")
    print(f"PubMedQA dropped invalid: {dropped}")
    print(f"PubMedQA after dedupe: {len(deduped)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
