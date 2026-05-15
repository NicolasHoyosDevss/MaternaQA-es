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
    find_first_non_empty,
    load_records_from_path,
    project_root,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Preprocess MedQuAD into chat JSONL format.")
    parser.add_argument("--input", type=Path, default=root / "data" / "raw" / "medquad")
    parser.add_argument("--output", type=Path, default=root / "data" / "processed" / "medquad_chat.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-dataset", type=str, default="lavita/MedQuAD")
    parser.add_argument("--hf-split", type=str, default="train")
    parser.add_argument("--use-local-first", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def extract_qa(record: Dict[str, Any]) -> tuple[str, str]:
    question = find_first_non_empty(record, ["question", "Question", "query", "input"])
    answer = find_first_non_empty(record, ["answer", "Answer", "output", "response"])
    return question, answer


def load_local_records(input_path: Path) -> List[Dict[str, Any]]:
    if not input_path.exists():
        return []
    return load_records_from_path(input_path)


def load_hf_records(dataset_name: str, split: str) -> Iterable[Dict[str, Any]]:
    ds = load_dataset(dataset_name, split=split)
    for row in ds:
        yield dict(row)


def main() -> None:
    args = parse_args()

    raw_records: List[Dict[str, Any]] = []
    if args.use_local_first:
        raw_records = load_local_records(args.input)

    if not raw_records:
        raw_records = list(load_hf_records(args.hf_dataset, args.hf_split))

    processed = []
    dropped = 0

    for record in tqdm(raw_records, desc="Processing MedQuAD"):
        question, answer = extract_qa(record)
        question = ensure_clean_non_empty(question)
        answer = clean_assistant_for_sft(answer, user_text=question, source="medquad")

        if not question or not answer:
            dropped += 1
            continue

        processed.append(build_chat_example(question, answer, source="medquad"))

    deduped = dedupe_by_pair(processed)
    write_jsonl(args.output, deduped)

    print(f"MedQuAD raw records: {len(raw_records)}")
    print(f"MedQuAD dropped invalid: {dropped}")
    print(f"MedQuAD after dedupe: {len(deduped)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
