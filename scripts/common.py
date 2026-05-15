from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import jsonlines
import pandas as pd

SYSTEM_PROMPT = (
    "You are a helpful medical AI assistant. "
    "Provide accurate and evidence-based medical information."
)
VALID_ROLES = {"system", "user", "assistant"}
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
MULTISPACE_RE = re.compile(r"\s+")

MEDQUAD_HPO_SENTENCE_RE = re.compile(
    r"The Human Phenotype Ontology provides the following list of signs and symptoms for .*?\.",
    re.IGNORECASE,
)
MEDQUAD_TABLE_SENTENCE_RE = re.compile(
    r"If the information is available, the table below includes how often the symptom is seen in people with this condition\.",
    re.IGNORECASE,
)
MEDQUAD_MEDLINE_SENTENCE_RE = re.compile(
    r"You can use the MedlinePlus Medical Dictionary to look up the definitions for these medical terms\.",
    re.IGNORECASE,
)
MEDQUAD_HPO_TAIL_RE = re.compile(
    r"The Human Phenotype Ontology \(HPO\) has collected information on how often a sign or symptom occurs in a condition\..*$",
    re.IGNORECASE | re.DOTALL,
)
MEDQUAD_TABLE_HEADER_RE = re.compile(
    r"Signs and Symptoms\s+Approximate number of patients \(when available\)",
    re.IGNORECASE,
)
FOOTERISH_LINE_RE = re.compile(
    r"(for more information|all rights reserved|copyright)",
    re.IGNORECASE,
)
HTML_TAG_SAFE_RE = re.compile(
    r"</?(?:b|i|em|strong|sup|sub|br|p|div|span|li|ul|ol|a)\b[^>]*>",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
SOURCE_MAX_WORDS = {"medquad": 260, "bioasq": 320, "pubmedqa": 180}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = CONTROL_CHARS_RE.sub("", text)

    normalized_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(normalized_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def build_chat_example(
    user_text: str,
    assistant_text: str,
    source: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"source": source}
    if extra_metadata:
        metadata.update(extra_metadata)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "metadata": metadata,
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        for row in rows:
            writer.write(row)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with jsonlines.open(path, mode="r") as reader:
        for row in reader:
            rows.append(row)
    return rows


def parse_json_file(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        for key in ("questions", "data", "examples", "items", "records"):
            maybe_list = data.get(key)
            if isinstance(maybe_list, list):
                return [x for x in maybe_list if isinstance(x, dict)]
        return [data]

    return []


def parse_jsonl_file(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with jsonlines.open(path, mode="r") as reader:
        for row in reader:
            if isinstance(row, dict):
                rows.append(row)
    return rows


def parse_tabular_file(path: Path) -> List[Dict[str, Any]]:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    return df.to_dict(orient="records")


def load_records_from_path(input_path: Path) -> List[Dict[str, Any]]:
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(
            p
            for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".csv", ".tsv"}
        )

    rows: List[Dict[str, Any]] = []
    for file in files:
        suffix = file.suffix.lower()
        if suffix == ".json":
            rows.extend(parse_json_file(file))
        elif suffix == ".jsonl":
            rows.extend(parse_jsonl_file(file))
        elif suffix in {".csv", ".tsv"}:
            rows.extend(parse_tabular_file(file))
    return rows


def dedupe_by_pair(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for ex in examples:
        try:
            user_text = ex["messages"][1]["content"]
            assistant_text = ex["messages"][2]["content"]
        except (IndexError, KeyError, TypeError):
            continue
        key = (user_text, assistant_text, ex.get("metadata", {}).get("source", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ex)
    return deduped


def validate_example(example: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    messages = example.get("messages")
    metadata = example.get("metadata")

    if not isinstance(messages, list) or len(messages) == 0:
        errors.append("messages_missing_or_empty")
        return errors

    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"message_{idx}_not_dict")
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in VALID_ROLES:
            errors.append(f"message_{idx}_invalid_role")
        if not is_non_empty_str(content):
            errors.append(f"message_{idx}_empty_content")

    source = metadata.get("source") if isinstance(metadata, dict) else None
    if not isinstance(metadata, dict) or not is_non_empty_str(source):
        errors.append("metadata_source_missing")

    return errors


def ensure_clean_non_empty(value: Any) -> str:
    return clean_text(value)


def _normalize_compare_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return MULTISPACE_RE.sub(" ", lowered).strip()


def _token_overlap(a: str, b: str) -> float:
    a_tokens = set(_normalize_compare_text(a).split())
    b_tokens = set(_normalize_compare_text(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens.intersection(b_tokens))
    return inter / max(1, len(b_tokens))


def _drop_question_restatement_prefix(answer: str, user_text: str) -> str:
    user_text = user_text.strip()
    if not user_text:
        return answer

    match = re.match(r"^\s*([^\n?.!]{1,300}\?)\s*", answer)
    if not match:
        return answer

    first_sentence = match.group(1).strip()
    if _token_overlap(first_sentence, user_text) >= 0.6:
        return answer[match.end() :].strip()
    return answer


def _dedupe_sentences(text: str) -> str:
    chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+|\n+", text) if chunk.strip()]
    if not chunks:
        return text

    seen = set()
    kept: List[str] = []
    for chunk in chunks:
        norm = _normalize_compare_text(chunk)
        if len(norm) >= 40 and norm in seen:
            continue
        if norm:
            seen.add(norm)
        kept.append(chunk)
    return " ".join(kept).strip()


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,;:-") + "."


def _strip_medquad_boilerplate(text: str) -> str:
    text = MEDQUAD_HPO_SENTENCE_RE.sub("", text)
    text = MEDQUAD_TABLE_SENTENCE_RE.sub("", text)
    text = MEDQUAD_MEDLINE_SENTENCE_RE.sub("", text)
    text = MEDQUAD_HPO_TAIL_RE.sub("", text)
    text = MEDQUAD_TABLE_HEADER_RE.sub("", text)
    return text


def _strip_footerish_lines(text: str) -> str:
    lines = [line.strip() for line in text.split("\n")]
    kept = [line for line in lines if line and not FOOTERISH_LINE_RE.search(line)]
    if not kept:
        return text
    return "\n".join(kept)


def clean_assistant_for_sft(answer: str, user_text: str, source: str) -> str:
    source_key = (source or "").strip().lower()
    text = clean_text(answer)
    text = html.unescape(text)
    text = HTML_TAG_SAFE_RE.sub(" ", text)
    text = URL_RE.sub("", text)
    text = clean_text(text)

    if source_key == "medquad":
        text = _strip_medquad_boilerplate(text)
        text = _drop_question_restatement_prefix(text, user_text)
        text = _strip_footerish_lines(text)

    if source_key == "bioasq":
        text = _dedupe_sentences(text)

    text = _dedupe_sentences(text)
    max_words = SOURCE_MAX_WORDS.get(source_key)
    if max_words:
        text = _truncate_words(text, max_words=max_words)

    return clean_text(text)


def normalize_weight_map(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return {k: v / total for k, v in weights.items()}


def find_first_non_empty(record: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        cleaned = clean_text(value)
        if cleaned:
            return cleaned
    return ""


def list_candidate_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted([p for p in path.rglob("*") if p.is_file()])
