#!/usr/bin/env python3
"""Evalúa datasets QA grounded con métricas estándar de Ragas.

Uso típico:
    python scripts/evaluate_qa_with_ragas.py --input datasets/obstetrics/qa/raw_C_gpt52_gen_gpt55_eval.jsonl

El script calcula Faithfulness y Answer Relevancy en dos pasadas internas y
genera un único reporte final. Se hace así porque en esta librería/entorno la
ejecución combinada puede quedarse colgada, mientras que por separado funcionó.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, field_validator


class CustomQuality(BaseModel):
    faithfulness: float
    answer_relevancy: float
    roundtrip_consistency: float
    verdict: str
    reason: str

    @field_validator("faithfulness", "answer_relevancy", "roundtrip_consistency", mode="before")
    @classmethod
    def clamp_score(cls, value: Any) -> float:
        score = float(value)
        return max(0.0, min(1.0, score))


ROUNDTRIP_SYSTEM_PROMPT = """\
Eres un asistente clínico. Responde con base en el contexto provisto.
Puedes usar conocimiento médico general únicamente para mejorar redacción y claridad,
pero no introduzcas hechos nuevos que no estén respaldados por el contexto.
Si algo no está suficientemente respaldado por el contexto, responde:
"No hay evidencia suficiente en el contexto."
"""

ROUNDTRIP_USER_TEMPLATE = """\
<contexto>
{context}
</contexto>

Pregunta:
{question}
"""

CUSTOM_JUDGE_SYSTEM_PROMPT = """\
Eres un evaluador estricto de calidad para datasets QA médicos.
Evalúa:
1) faithfulness: qué tan respaldada está la respuesta por el contexto.
2) answer_relevancy: qué tan bien responde la pregunta.
3) roundtrip_consistency: qué tan consistente es con una segunda respuesta independiente.
Retorna puntajes [0,1], verdict ("accept" o "reject") y reason breve.
"""

CUSTOM_JUDGE_USER_TEMPLATE = """\
Contexto:
{context}

Pregunta:
{question}

Respuesta original:
{answer}

Respuesta roundtrip:
{roundtrip_answer}
"""

META_REFERENCE_PATTERNS = [
    "según el texto",
    "según el fragmento",
    "según la tabla",
    "de acuerdo con el fragmento",
    "de acuerdo al fragmento",
    "con base en el texto",
    "basado en el texto",
    "el fragmento dice",
    "el contexto señala",
    "el contexto compartido",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula Faithfulness y Answer Relevancy con Ragas real."
    )
    parser.add_argument("--input", type=Path, required=True, help="Archivo raw_*.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Reporte JSON de salida. Si se omite, se crea eval_<nombre_input>.json en la misma carpeta.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evalúa solo los primeros N pares (útil para validar rápido).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Evalúa una muestra aleatoria estratificada por source_pdf. Recomendado para dataset final.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad cuando se usa --sample-size.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="Timeout máximo por métrica/par.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="Modelo evaluador usado por Ragas.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="Modelo de embeddings para Answer Relevancy.",
    )
    parser.add_argument(
        "--custom-judge-model",
        type=str,
        default=None,
        help=(
            "Opcional. Si se define, además de Ragas calcula la validación custom "
            "sobre la misma muestra usando este modelo."
        ),
    )
    return parser.parse_args()


def stratified_sample(
    rows: List[Dict[str, Any]],
    sample_size: int | None,
    seed: int,
) -> List[Dict[str, Any]]:
    """Muestra reproducible, intentando no dejar que un PDF domine la evaluación."""
    if sample_size is None or sample_size >= len(rows):
        return rows
    if sample_size <= 0:
        return []

    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("source_pdf") or "unknown")
        groups.setdefault(key, []).append(row)

    for group_rows in groups.values():
        rng.shuffle(group_rows)

    total = len(rows)
    quotas = []
    for key, group_rows in groups.items():
        raw_quota = sample_size * len(group_rows) / total
        base = int(raw_quota)
        if base == 0 and len(groups) <= sample_size:
            base = 1
        quotas.append((key, base, raw_quota - int(raw_quota)))

    selected: List[Dict[str, Any]] = []
    for key, quota, _frac in quotas:
        selected.extend(groups[key][:quota])

    remaining = sample_size - len(selected)
    if remaining > 0:
        used_ids = {id(row) for row in selected}
        leftovers = [row for group_rows in groups.values() for row in group_rows if id(row) not in used_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[:remaining])
    elif remaining < 0:
        rng.shuffle(selected)
        selected = selected[:sample_size]

    rng.shuffle(selected)
    return selected


def has_meta_reference(text: str) -> bool:
    lower = str(text).lower()
    return any(pattern in lower for pattern in META_REFERENCE_PATTERNS)


def basic_cleanliness(row: Dict[str, Any]) -> Dict[str, Any]:
    question = str(row.get("pregunta", ""))
    answer = str(row.get("respuesta", ""))
    return {
        "question_has_meta_reference": has_meta_reference(question),
        "answer_has_meta_reference": has_meta_reference(answer),
        "question_mark_count": question.count("?") + question.count("¿"),
        "answer_word_count": len(answer.split()),
    }


def extract_metric_value(result: Any) -> Any:
    """Intenta extraer un valor numérico de distintos formatos de resultado."""
    if isinstance(result, (int, float)):
        return float(result)
    if isinstance(result, dict):
        return result.get("value")
    return getattr(result, "value", None)


def extract_metric_reason(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("reason", "") or "")
    return str(getattr(result, "reason", "") or "")


async def evaluate_rows(
    rows: List[Dict[str, Any]],
    llm_model: str,
    embedding_model: str,
    timeout_seconds: int,
    metric: str,
) -> List[Dict[str, Any]]:
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy, Faithfulness
    except ImportError as exc:
        raise SystemExit(
            "Faltan dependencias para Ragas. Ejecuta: pip install -r requirements.txt"
        ) from exc

    client = AsyncOpenAI()
    llm = llm_factory(llm_model, client=client)
    embeddings = embedding_factory("openai", model=embedding_model, client=client)
    faithfulness = Faithfulness(llm=llm)
    relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)

    results: List[Dict[str, Any]] = []
    total = len(rows)
    for i, row in enumerate(rows, start=1):
        try:
            contexts = [row["contexto_fuente"]]
            faith_value = None
            rel_value = None
            faith_reason = ""
            rel_reason = ""
            if metric == "faithfulness":
                print(f"[RAGAS] par {i}/{total} -> faithfulness...", flush=True)
                faith = await asyncio.wait_for(
                    faithfulness.ascore(
                        user_input=row["pregunta"],
                        response=row["respuesta"],
                        retrieved_contexts=contexts,
                    ),
                    timeout=timeout_seconds,
                )
                faith_value = extract_metric_value(faith)
                faith_reason = extract_metric_reason(faith)
                if faith_value is None:
                    raise RuntimeError(f"Ragas devolvió faithfulness vacío. reason={faith_reason}")
            elif metric == "answer_relevancy":
                print(f"[RAGAS] par {i}/{total} -> answer_relevancy...", flush=True)
                rel = await asyncio.wait_for(
                    relevancy.ascore(
                        user_input=row["pregunta"],
                        response=row["respuesta"],
                    ),
                    timeout=timeout_seconds,
                )
                rel_value = extract_metric_value(rel)
                rel_reason = extract_metric_reason(rel)
                if rel_value is None:
                    raise RuntimeError(f"Ragas devolvió answer_relevancy vacío. reason={rel_reason}")
            else:
                raise ValueError(f"Métrica no soportada: {metric}")
            results.append(
                {
                    "qa_id": row["qa_id"],
                    "chunk_id": row["chunk_id"],
                    "ragas_faithfulness": float(faith_value) if faith_value is not None else None,
                    "ragas_answer_relevancy": float(rel_value) if rel_value is not None else None,
                    "ragas_faithfulness_reason": faith_reason,
                    "ragas_answer_relevancy_reason": rel_reason,
                }
            )
        except Exception as exc:
            error_row = {
                "qa_id": row.get("qa_id"),
                "chunk_id": row.get("chunk_id"),
                "ragas_faithfulness": None,
                "ragas_answer_relevancy": None,
                "error_type": type(exc).__name__,
                "error": repr(exc),
            }
            results.append(error_row)
        print(f"[RAGAS] {i}/{total} pares evaluados", flush=True)
    return results


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    faith = [r["ragas_faithfulness"] for r in results if r.get("ragas_faithfulness") is not None]
    rel = [r["ragas_answer_relevancy"] for r in results if r.get("ragas_answer_relevancy") is not None]
    custom_faith = [r["custom_faithfulness"] for r in results if r.get("custom_faithfulness") is not None]
    custom_rel = [r["custom_answer_relevancy"] for r in results if r.get("custom_answer_relevancy") is not None]
    custom_rt = [r["custom_roundtrip_consistency"] for r in results if r.get("custom_roundtrip_consistency") is not None]
    errors = sum(1 for r in results if "error" in r)
    return {
        "pairs_evaluated": len(results),
        "pairs_scored_faithfulness": len(faith),
        "pairs_scored_answer_relevancy": len(rel),
        "pairs_with_error": errors,
        "avg_ragas_faithfulness": round(statistics.mean(faith), 4) if faith else None,
        "min_ragas_faithfulness": round(min(faith), 4) if faith else None,
        "max_ragas_faithfulness": round(max(faith), 4) if faith else None,
        "avg_ragas_answer_relevancy": round(statistics.mean(rel), 4) if rel else None,
        "min_ragas_answer_relevancy": round(min(rel), 4) if rel else None,
        "max_ragas_answer_relevancy": round(max(rel), 4) if rel else None,
        "custom_pairs_scored": len(custom_faith),
        "custom_acceptance_rate": (
            round(sum(1 for r in results if r.get("custom_verdict") == "accept") / len(custom_faith), 4)
            if custom_faith
            else None
        ),
        "avg_custom_faithfulness": round(statistics.mean(custom_faith), 4) if custom_faith else None,
        "avg_custom_answer_relevancy": round(statistics.mean(custom_rel), 4) if custom_rel else None,
        "avg_custom_roundtrip_consistency": round(statistics.mean(custom_rt), 4) if custom_rt else None,
        "questions_with_meta_reference": sum(1 for r in results if r.get("question_has_meta_reference")),
        "answers_with_meta_reference": sum(1 for r in results if r.get("answer_has_meta_reference")),
    }


async def evaluate_custom_quality(
    rows: List[Dict[str, Any]],
    model: str,
    timeout_seconds: int,
) -> List[Dict[str, Any]]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    results: List[Dict[str, Any]] = []
    total = len(rows)
    for i, row in enumerate(rows, start=1):
        print(f"[CUSTOM] par {i}/{total} -> quality judge...", flush=True)
        try:
            context = str(row.get("contexto_fuente") or "")
            roundtrip_response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": ROUNDTRIP_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": ROUNDTRIP_USER_TEMPLATE.format(
                                context=context,
                                question=row.get("pregunta", ""),
                            ),
                        },
                    ],
                ),
                timeout=timeout_seconds,
            )
            roundtrip_answer = (roundtrip_response.choices[0].message.content or "").strip()

            judge_response = await asyncio.wait_for(
                client.chat.completions.parse(
                    model=model,
                    messages=[
                        {"role": "system", "content": CUSTOM_JUDGE_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": CUSTOM_JUDGE_USER_TEMPLATE.format(
                                context=context,
                                question=row.get("pregunta", ""),
                                answer=row.get("respuesta", ""),
                                roundtrip_answer=roundtrip_answer,
                            ),
                        },
                    ],
                    response_format=CustomQuality,
                ),
                timeout=timeout_seconds,
            )
            parsed = judge_response.choices[0].message.parsed
            if parsed is None:
                raise RuntimeError("custom_judge_parse_none")
            results.append(
                {
                    "qa_id": row.get("qa_id"),
                    "custom_faithfulness": parsed.faithfulness,
                    "custom_answer_relevancy": parsed.answer_relevancy,
                    "custom_roundtrip_consistency": parsed.roundtrip_consistency,
                    "custom_verdict": parsed.verdict,
                    "custom_reason": parsed.reason,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "qa_id": row.get("qa_id"),
                    "custom_faithfulness": None,
                    "custom_answer_relevancy": None,
                    "custom_roundtrip_consistency": None,
                    "custom_verdict": None,
                    "custom_reason": "",
                    "custom_error_type": type(exc).__name__,
                    "custom_error": repr(exc),
                }
            )
    return results


async def main_async(args: argparse.Namespace) -> None:
    all_rows = read_jsonl(args.input)
    rows = stratified_sample(all_rows, args.sample_size, args.seed)
    if args.limit is not None:
        rows = rows[: args.limit]
    print(
        f"[EVAL] Iniciando evaluación de {len(rows)} pares "
        f"(dataset completo: {len(all_rows)})",
        flush=True,
    )

    # Implementación simple para el usuario: una sola ejecución.
    # Implementación robusta por dentro: dos pasadas y merge final.
    print("[RAGAS] Calculando faithfulness...", flush=True)
    faith_rows = await evaluate_rows(
        rows,
        args.llm_model,
        args.embedding_model,
        args.timeout_seconds,
        metric="faithfulness",
    )
    print("[RAGAS] Calculando answer_relevancy...", flush=True)
    rel_rows = await evaluate_rows(
        rows,
        args.llm_model,
        args.embedding_model,
        args.timeout_seconds,
        metric="answer_relevancy",
    )

    faith_by_id = {r.get("qa_id"): r for r in faith_rows}
    rel_by_id = {r.get("qa_id"): r for r in rel_rows}

    custom_by_id: Dict[str, Dict[str, Any]] = {}
    if args.custom_judge_model:
        print(f"[CUSTOM] Calculando validación custom con {args.custom_judge_model}...", flush=True)
        custom_rows = await evaluate_custom_quality(rows, args.custom_judge_model, args.timeout_seconds)
        custom_by_id = {r.get("qa_id"): r for r in custom_rows}

    results = []
    for row in rows:
        qa_id = row.get("qa_id")
        f = faith_by_id.get(qa_id, {})
        r = rel_by_id.get(qa_id, {})
        c = custom_by_id.get(qa_id, {})
        merged = {
            "qa_id": qa_id,
            "chunk_id": row.get("chunk_id"),
            "source_pdf": row.get("source_pdf"),
            "tipo": row.get("tipo"),
            "dificultad": row.get("dificultad"),
            "pregunta": row.get("pregunta"),
            "respuesta": row.get("respuesta"),
            "ragas_faithfulness": f.get("ragas_faithfulness"),
            "ragas_answer_relevancy": r.get("ragas_answer_relevancy"),
            "ragas_faithfulness_reason": f.get("ragas_faithfulness_reason", ""),
            "ragas_answer_relevancy_reason": r.get("ragas_answer_relevancy_reason", ""),
            "custom_faithfulness": c.get("custom_faithfulness"),
            "custom_answer_relevancy": c.get("custom_answer_relevancy"),
            "custom_roundtrip_consistency": c.get("custom_roundtrip_consistency"),
            "custom_verdict": c.get("custom_verdict"),
            "custom_reason": c.get("custom_reason", ""),
            **basic_cleanliness(row),
        }
        errs = []
        if f.get("error"):
            errs.append(f"faithfulness: {f.get('error_type','Error')} {f.get('error')}")
        if r.get("error"):
            errs.append(f"answer_relevancy: {r.get('error_type','Error')} {r.get('error')}")
        if c.get("custom_error"):
            errs.append(f"custom_judge: {c.get('custom_error_type','Error')} {c.get('custom_error')}")
        if errs:
            merged["error_type"] = "MetricPassError"
            merged["error"] = " | ".join(errs)
        results.append(merged)

    output = args.output
    if output is None:
        prefix = f"eval_sample{len(rows)}_" if args.sample_size else "eval_"
        output = args.input.with_name(f"{prefix}{args.input.stem}.json")

    report = {
        "script_version": "qa_sample_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_file": str(args.input),
        "total_pairs_in_input": len(all_rows),
        "sampled_pairs": len(rows),
        "sample_size_requested": args.sample_size,
        "sample_seed": args.seed if args.sample_size else None,
        "execution": "single_command_two_internal_passes",
        "ragas_models": {
            "llm_model": args.llm_model,
            "embedding_model": args.embedding_model,
        },
        "custom_judge_model": args.custom_judge_model,
        "summary": summarize(results),
        "per_pair": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[RAGAS] Reporte guardado en: {output}", flush=True)


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
