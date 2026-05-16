"""
generate_synthetic_qa.py
========================
Genera pares sintéticos de Pregunta/Respuesta en español a partir de los
chunks ya procesados en data/obstetrics_spanish/train_lm.jsonl.

Usa OpenAI Structured Outputs (gpt-4o-mini / gpt-4o) para garantizar que
la salida sea un JSON válido y alineado con el esquema Pydantic. Todos los
registros se escriben en formato messages (SFT-ready) para Unsloth o
HuggingFace TRL.

Flujo:
  1. Lee los chunks de train_lm.jsonl (producidos por build_obstetrics_lm_dataset.py)
  2. Determina cuántos pares generar por chunk según token_estimate y clinical_score
  3. Llama a la API de forma asíncrona (semáforo configurable)
  4. Guarda checkpoint incremental para poder reanudar si algo falla
  5. Escribe dos archivos:
       - synthetic_qa_raw.jsonl  → pares crudos con metadatos para auditoría
       - synthetic_qa_sft.jsonl  → formato {"messages": [...], "metadata": {...}}
                                   listo para entrenamiento SFT/QLoRA

Uso básico:
    export OPENAI_API_KEY="sk-..."
    python scripts/obstetrics/generate_synthetic_qa.py

Dry-run (sin llamar a la API):
    python scripts/obstetrics/generate_synthetic_qa.py --dry-run

Estimación de costo (856 chunks, mayo 2026):
    gpt-5.4-mini  ≈ $1.10  USD  (recomendado para presupuesto)
    gpt-5.4       ≈ $4.50  USD  (balance calidad/costo)
    gpt-5.5       ≈ $8.00  USD  (máxima calidad clínica)

Nota: gpt-4o y gpt-4o-mini fueron deprecados en febrero 2026.

Versiones mínimas requeridas:
    openai>=1.68.0   (structured outputs estables, sin .beta.)
    pydantic>=2.7.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from pydantic import BaseModel, field_validator
from tqdm.asyncio import tqdm as atqdm

# ---------------------------------------------------------------------------
# Pydantic: esquema de respuesta (Structured Outputs de OpenAI)
# ---------------------------------------------------------------------------

TipoPregunta = Literal[
    "factual",
    "razonamiento",
    "definicion",
    "comparacion",
    "aplicacion",
    "hipotetico",
]
NivelDificultad = Literal["basico", "intermedio", "avanzado"]


class QAPar(BaseModel):
    pregunta: str
    respuesta: str
    tipo: TipoPregunta
    dificultad: NivelDificultad
    contexto_fuente: str

    @field_validator("pregunta", "respuesta", "contexto_fuente", mode="before")
    @classmethod
    def no_empty(cls, v: Any) -> str:
        text = str(v).strip()
        if not text:
            raise ValueError("El campo no puede estar vacío.")
        return text


class RespuestaGeneracion(BaseModel):
    pares: List[QAPar]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Prompt del sistema usado DENTRO del mensaje SFT (lo que verá el modelo al inferir).
SFT_SYSTEM_PROMPT = (
    "Eres un asistente médico especializado en obstetricia y ginecología. "
    "Responde en español con precisión clínica, usando vocabulario médico apropiado. "
    "Tus respuestas deben ser claras, basadas en evidencia y coherentes con las guías "
    "y protocolos clínicos vigentes en Latinoamérica. Cuando la pregunta implique una "
    "situación de urgencia obstétrica, indica explícitamente que requiere atención "
    "médica inmediata."
)

# Prompt del sistema que le damos a GPT para que GENERE los pares.
GENERATION_SYSTEM_PROMPT = """\
Eres un experto en generación de datasets de entrenamiento para modelos de lenguaje.
Tu tarea es leer fragmentos de una base de conocimiento médica en obstetricia y ginecología,
y generar pares de pregunta-respuesta de alta calidad en español para fine-tuning
supervisado (SFT).

Reglas estrictas:
1. Las preguntas deben ser DIVERSAS en tipo: factuales, de razonamiento, de definición,
   de comparación, de aplicación práctica y de "qué pasa si" (hipotético).
2. Las respuestas deben ser completas, precisas y basadas ÚNICAMENTE en el texto dado.
   No inventes información que no esté en el contexto.
3. Usa español natural y vocabulario médico correcto; no traduzcas literalmente del inglés.
4. Varía la longitud de las respuestas: algunas cortas (1-2 oraciones) y otras desarrolladas
   (varios párrafos si el tema lo amerita).
5. El campo "contexto_fuente" debe ser un fragmento breve (máximo 2 oraciones del texto)
   que respalda directamente la respuesta.
6. Distribuye los tipos a lo largo de los pares: no repitas el mismo tipo consecutivamente
   si tienes más de 2 pares.
7. Para preguntas de tipo "aplicacion" usa viñetas clínicas cortas (paciente con X condición).
"""

# Template del mensaje de usuario enviado a GPT por cada chunk.
GENERATION_USER_TEMPLATE = """\
Documento fuente: {source_pdf}
Sección: {section}

<contexto>
{text}
</contexto>

Genera exactamente {n_pairs} pares de pregunta-respuesta en español basados en el \
fragmento anterior.
"""

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MAX_RETRIES = 5
BASE_BACKOFF_S = 2.0

# Modelos activos en la API de OpenAI a mayo 2026.
# gpt-4o y gpt-4o-mini fueron deprecados en febrero 2026.
SUPPORTED_MODELS = (
    "gpt-5.4-mini",  # Más barato y rápido, reemplaza a gpt-4o-mini
    "gpt-5.4",  # Balance calidad/costo
    "gpt-5.5",  # Flagship, mejor calidad, más caro
)

# Precios por millón de tokens (mayo 2026, fuente: platform.openai.com/api/docs/models)
PRICES_PER_M = {
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5.5": {"input": 5.00, "output": 30.00},
}

# Estimación conservadora de tokens de respuesta por par generado
TOKENS_PER_PAIR_OUTPUT_EST = 160


# ---------------------------------------------------------------------------
# Helpers de I/O
# ---------------------------------------------------------------------------


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    return project_root() / "data" / "obstetrics_spanish"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Escribe los registros en modo append para no perder datos si el script se interrumpe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Checkpoint / progress
# ---------------------------------------------------------------------------


def load_progress(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("processed_chunk_ids", []))


def save_progress(path: Path, processed_ids: Set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"processed_chunk_ids": sorted(processed_ids)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Lógica de pares por chunk
# ---------------------------------------------------------------------------


def n_pairs_for_chunk(
    token_estimate: int,
    clinical_score: int,
    min_pairs: int,
    max_pairs: int,
) -> int:
    """
    Determina cuántos pares generar basándose en el tamaño y relevancia clínica del chunk.

    Lógica:
    - Chunks pequeños (<400 tokens)  → mínimo de pares.
    - Chunks medianos (400-700)      → mínimo + 1.
    - Chunks grandes (>700 tokens)   → máximo - 1.
    - clinical_score >= 20           → +1 par extra (hasta el máximo).
    """
    if token_estimate < 400:
        n = min_pairs
    elif token_estimate < 700:
        n = min_pairs + 1
    else:
        n = max(min_pairs + 1, max_pairs - 1)

    if clinical_score >= 20:
        n = min(n + 1, max_pairs)

    return min(max(n, min_pairs), max_pairs)


def estimate_cost(
    chunks: List[Dict[str, Any]],
    model: str,
    min_pairs: int,
    max_pairs: int,
) -> Tuple[int, float]:
    """Devuelve (pares_esperados, costo_estimado_usd)."""
    expected_pairs = sum(
        n_pairs_for_chunk(
            c.get("metadata", {}).get("token_estimate", 500),
            c.get("metadata", {}).get("clinical_score", 0),
            min_pairs,
            max_pairs,
        )
        for c in chunks
    )
    total_input_tokens = sum(
        c.get("metadata", {}).get("token_estimate", 500) for c in chunks
    )
    total_output_tokens = expected_pairs * TOKENS_PER_PAIR_OUTPUT_EST

    prices = PRICES_PER_M.get(model, PRICES_PER_M["gpt-5.4-mini"])
    cost = (total_input_tokens / 1_000_000) * prices["input"] + (
        total_output_tokens / 1_000_000
    ) * prices["output"]
    return expected_pairs, cost


# ---------------------------------------------------------------------------
# Conversión a formato SFT
# ---------------------------------------------------------------------------


def chunk_to_sft_records(
    chunk: Dict[str, Any],
    pairs: List[QAPar],
) -> List[Dict[str, Any]]:
    """Convierte los pares generados al formato messages listo para fine-tuning."""
    meta = chunk.get("metadata", {})
    records = []
    for idx, pair in enumerate(pairs, start=1):
        chunk_id = str(meta.get("chunk_id", ""))
        qa_id = f"{chunk_id}_qa_{idx:03d}" if chunk_id else f"qa_{idx:03d}"
        records.append(
            {
                "messages": [
                    {"role": "system", "content": SFT_SYSTEM_PROMPT},
                    {"role": "user", "content": pair.pregunta},
                    {"role": "assistant", "content": pair.respuesta},
                ],
                "metadata": {
                    "source": "obstetrics_spanish_synthetic",
                    "source_pdf": meta.get("source_pdf", ""),
                    "chunk_id": meta.get("chunk_id", ""),
                    "qa_id": qa_id,
                    "pages": meta.get("pages", []),
                    "section": meta.get("section", ""),
                    "section_type": meta.get("section_type", ""),
                    "content_role": meta.get("content_role", ""),
                    "topics": meta.get("topics", []) or meta.get("topic_tags", []),
                    "split": meta.get("split", ""),
                    "clinical_score": meta.get("clinical_score", 0),
                    "token_estimate": meta.get("token_estimate", 0),
                    "tipo": pair.tipo,
                    "dificultad": pair.dificultad,
                    "contexto_fuente": pair.contexto_fuente,
                },
            }
        )
    return records


def chunk_to_raw_records(
    chunk: Dict[str, Any],
    pairs: List[QAPar],
) -> List[Dict[str, Any]]:
    """Guarda los pares en formato plano para auditoría / revisión humana."""
    meta = chunk.get("metadata", {})
    rows: List[Dict[str, Any]] = []
    chunk_id = str(meta.get("chunk_id", ""))
    for idx, p in enumerate(pairs, start=1):
        qa_id = f"{chunk_id}_qa_{idx:03d}" if chunk_id else f"qa_{idx:03d}"
        rows.append(
            {
                "qa_id": qa_id,
                "chunk_id": meta.get("chunk_id", ""),
                "source_pdf": meta.get("source_pdf", ""),
                "section": meta.get("section", ""),
                "section_type": meta.get("section_type", ""),
                "content_role": meta.get("content_role", ""),
                "topics": meta.get("topics", []) or meta.get("topic_tags", []),
                "split": meta.get("split", ""),
                "pages": meta.get("pages", []),
                "clinical_score": meta.get("clinical_score", 0),
                "token_estimate": meta.get("token_estimate", 0),
                "pregunta": p.pregunta,
                "respuesta": p.respuesta,
                "tipo": p.tipo,
                "dificultad": p.dificultad,
                "contexto_fuente": p.contexto_fuente,
            }
        )
    return rows


def _norm_tokens(text: str) -> Set[str]:
    return set(re.findall(r"\b\w+\b", str(text).lower(), flags=re.UNICODE))


def grounding_metrics_for_pairs(pairs: List[QAPar]) -> Dict[str, Any]:
    overlap_ratios: List[float] = []
    low_grounding = 0
    for pair in pairs:
        ctx = _norm_tokens(pair.contexto_fuente)
        ans = _norm_tokens(pair.respuesta)
        if not ctx:
            overlap_ratios.append(0.0)
            low_grounding += 1
            continue
        overlap = len(ctx & ans) / max(1, len(ctx))
        overlap_ratios.append(overlap)
        if overlap < 0.15:
            low_grounding += 1
    avg_overlap = sum(overlap_ratios) / max(1, len(overlap_ratios))
    return {
        "avg_context_answer_overlap": round(avg_overlap, 4),
        "low_grounding_pairs": low_grounding,
        "total_pairs": len(pairs),
    }


# ---------------------------------------------------------------------------
# Generación asíncrona
# ---------------------------------------------------------------------------


async def generate_for_chunk(
    client: Any,  # AsyncOpenAI
    chunk: Dict[str, Any],
    model: str,
    min_pairs: int,
    max_pairs: int,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], List[QAPar], str]:
    """Llama a la API para un único chunk. Incluye reintentos con backoff exponencial."""
    meta = chunk.get("metadata", {})
    chunk_id = meta.get("chunk_id", "unknown")
    n = n_pairs_for_chunk(
        meta.get("token_estimate", 500),
        meta.get("clinical_score", 0),
        min_pairs,
        max_pairs,
    )

    user_content = GENERATION_USER_TEMPLATE.format(
        source_pdf=meta.get("source_pdf", "documento"),
        section=meta.get("section", "sin sección") or "sin sección",
        text=chunk.get("text", "").strip(),
        n_pairs=n,
    )

    backoff = BASE_BACKOFF_S
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with semaphore:
                response = await client.chat.completions.parse(
                    model=model,
                    messages=[
                        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    response_format=RespuestaGeneracion,
                    temperature=0.75,
                )

            # Verificar rechazo del modelo
            choice = response.choices[0]
            if getattr(choice.message, "refusal", None):
                logger.warning(
                    "Chunk %s fue rechazado por el modelo. Saltando.", chunk_id
                )
                return chunk, [], "refused"

            parsed = choice.message.parsed
            if parsed is None:
                raise ValueError("Structured output devolvió None.")

            return chunk, parsed.pares, "ok"

        except Exception as exc:
            # Importar aquí para evitar dependencia en el top-level (puede no estar instalado aún)
            try:
                from openai import APIStatusError, RateLimitError
            except ImportError:
                RateLimitError = None  # type: ignore
                APIStatusError = None  # type: ignore

            is_rate_limit = RateLimitError and isinstance(exc, RateLimitError)
            is_server_error = (
                APIStatusError
                and isinstance(exc, APIStatusError)
                and exc.status_code is not None
                and exc.status_code >= 500
            )

            if attempt == MAX_RETRIES:
                logger.error(
                    "Chunk %s: máximo de reintentos alcanzado (%d). Error: %s",
                    chunk_id,
                    MAX_RETRIES,
                    exc,
                )
                return chunk, [], "failed"

            if is_rate_limit or is_server_error:
                wait = backoff * (2 ** (attempt - 1)) + random.uniform(0, 1)
                logger.warning(
                    "Chunk %s: %s. Reintento %d/%d en %.1fs.",
                    chunk_id,
                    "Rate limit" if is_rate_limit else f"Error servidor ({exc})",
                    attempt,
                    MAX_RETRIES,
                    wait,
                )
                await asyncio.sleep(wait)
            else:
                logger.error("Chunk %s: error no recuperable: %s", chunk_id, exc)
                return chunk, [], "failed"

    return chunk, [], "failed"


async def run_generation(
    chunks: List[Dict[str, Any]],
    client: Any,
    model: str,
    min_pairs: int,
    max_pairs: int,
    concurrency: int,
    sft_output: Path,
    raw_output: Path,
    progress_file: Path,
    report_output: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    processed_ids = load_progress(progress_file)

    # Solo procesar lo que no se haya procesado aún (resume-safe)
    pending = [
        c
        for c in chunks
        if c.get("metadata", {}).get("chunk_id", "") not in processed_ids
    ]

    logger.info(
        "Chunks totales: %d | Ya procesados: %d | Pendientes: %d",
        len(chunks),
        len(processed_ids),
        len(pending),
    )

    if not pending:
        logger.info("Todos los chunks ya fueron procesados.")
        return {
            "total": len(chunks),
            "skipped": len(chunks),
            "processed": 0,
            "failed": 0,
            "qa_pairs": 0,
        }

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        generate_for_chunk(
            client, chunk, model, min_pairs, max_pairs, semaphore, logger
        )
        for chunk in pending
    ]

    processed = 0
    failed = 0
    total_pairs = 0
    grounding_overlap_sum = 0.0
    grounding_pairs = 0
    low_grounding_pairs = 0

    # as_completed para poder escribir y actualizar checkpoint en cuanto cada tarea termina
    for coro in atqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Generando QA"
    ):
        chunk, pairs, status = await coro
        chunk_id = chunk.get("metadata", {}).get("chunk_id", "unknown")

        if pairs:
            append_jsonl(sft_output, chunk_to_sft_records(chunk, pairs))
            append_jsonl(raw_output, chunk_to_raw_records(chunk, pairs))
            gm = grounding_metrics_for_pairs(pairs)
            grounding_overlap_sum += float(gm["avg_context_answer_overlap"]) * int(gm["total_pairs"])
            grounding_pairs += int(gm["total_pairs"])
            low_grounding_pairs += int(gm["low_grounding_pairs"])
            processed += 1
            total_pairs += len(pairs)
            processed_ids.add(chunk_id)
            save_progress(progress_file, processed_ids)
        elif status == "refused":
            failed += 1
            processed_ids.add(chunk_id)
            save_progress(progress_file, processed_ids)
        else:
            failed += 1

    stats = {
        "total": len(chunks),
        "skipped": len(chunks) - len(pending),
        "processed": processed,
        "failed": failed,
        "qa_pairs": total_pairs,
        "grounding": {
            "avg_context_answer_overlap": round(grounding_overlap_sum / max(1, grounding_pairs), 4),
            "low_grounding_pairs": low_grounding_pairs,
            "total_pairs": grounding_pairs,
            "low_grounding_rate": round(low_grounding_pairs / max(1, grounding_pairs), 4),
        },
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    data_dir = default_data_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Genera pares sintéticos QA en español desde los chunks de obstetricia "
            "usando OpenAI Structured Outputs. Salida en formato SFT (messages)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=data_dir / "train_lm.jsonl",
        help="JSONL de chunks limpios (train_lm.jsonl o validation_lm.jsonl).",
    )
    parser.add_argument(
        "--sft-output",
        type=Path,
        default=data_dir / "synthetic_qa_sft.jsonl",
        help="Salida en formato messages, lista para SFT/QLoRA.",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=data_dir / "synthetic_qa_raw.jsonl",
        help="Salida de auditoría con los pares crudos y metadatos.",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=data_dir / ".qa_generation_progress.json",
        help="Archivo de checkpoint para reanudar si el proceso se interrumpe.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=data_dir / "qa_generation_report.json",
        help="Reporte de métricas de generación y grounding.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4-mini",
        choices=SUPPORTED_MODELS,
        help=(
            "Modelo de OpenAI a utilizar. "
            "gpt-5.4-mini es el más económico; "
            "gpt-5.4 para mejor calidad; "
            "gpt-5.5 para máxima calidad (más caro). "
            "Nota: gpt-4o y gpt-4o-mini fueron deprecados en feb 2026."
        ),
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=2,
        help="Número mínimo de pares QA por chunk.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=5,
        help="Número máximo de pares QA por chunk.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=15,
        help="Peticiones simultáneas a la API de OpenAI.",
    )
    parser.add_argument(
        "--min-clinical-score",
        type=int,
        default=0,
        help="Ignorar chunks con clinical_score menor a este valor.",
    )
    # Phase 7: content-role filtering for clinically useful QA generation
    CLINICAL_ROLES = ("evidence", "recommendation", "procedure", "diagnostic", "treatment")
    parser.add_argument(
        "--allowed-content-roles",
        type=str,
        default=",".join(CLINICAL_ROLES),
        help="Roles de contenido permitidos (coma-separados). Fase 7: solo roles clínicamente útiles.",
    )
    parser.add_argument(
        "--no-content-role-filter",
        action="store_true",
        help="Deshabilitar el filtro por content_role (backward-compatible).",
    )
    parser.add_argument(
        "--min-topic-coverage",
        type=float,
        default=0.30,
        help="Fase 7: reporta cobertura mínima por topic sin reinyectar roles no accionables (0.0-1.0).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key de OpenAI. Por defecto usa la variable OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra estadísticas y estimación de costo sin llamar a la API.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Procesa solo los primeros N chunks tras los filtros. Útil para pruebas pequeñas.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Phase 7: content-role filtering with topic-coverage preservation
# ---------------------------------------------------------------------------


CLINICAL_CONTENT_ROLES = {
    "evidence",
    "recommendation",
    "procedure",
    "diagnostic",
    "treatment",
}


def compute_topic_distribution(chunks: List[Dict[str, Any]]) -> Counter[str]:
    """Compute topic frequency across chunks."""
    dist: Counter[str] = Counter()
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        topics = meta.get("topics", []) or meta.get("topic_tags", [])
        for topic in topics:
            if topic:
                dist[topic] += 1
    return dist


def chunk_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    metadata = chunk.get("metadata")
    return metadata if isinstance(metadata, dict) else chunk


def chunk_content_role(chunk: Dict[str, Any]) -> str:
    return str(chunk_metadata(chunk).get("content_role", "")).strip()


def filter_by_content_role(
    chunks: List[Dict[str, Any]],
    allowed_roles: Set[str],
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Filter chunks to only actionable content roles and report skipped stats."""
    before = len(chunks)
    filtered: List[Dict[str, Any]] = []
    kept_by_role: Counter[str] = Counter()
    skipped_by_role: Counter[str] = Counter()
    for chunk in chunks:
        role = chunk_content_role(chunk) or "unknown"
        if role in allowed_roles:
            filtered.append(chunk)
            kept_by_role[role] += 1
        else:
            skipped_by_role[role] += 1
    logger.info(
        "Filtro content_role %s: %d → %d chunks",
        allowed_roles,
        before,
        len(filtered),
    )
    if skipped_by_role:
        logger.info(
            "Chunks excluidos por content_role: %s",
            dict(sorted(skipped_by_role.items())),
        )
    return filtered, {
        "input_chunks": before,
        "eligible_chunks": len(filtered),
        "skipped_chunks": sum(skipped_by_role.values()),
        "eligible_by_content_role": dict(sorted(kept_by_role.items())),
        "skipped_by_content_role": dict(sorted(skipped_by_role.items())),
        "allowed_content_roles": sorted(allowed_roles),
    }


def preserve_topic_coverage(
    original: List[Dict[str, Any]],
    filtered: List[Dict[str, Any]],
    min_coverage: float,
    allowed_roles: Set[str],
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """Report topic coverage without relaxing the actionable-role filter."""
    if min_coverage <= 0.0:
        return filtered

    original_dist = compute_topic_distribution(original)
    if not original_dist:
        return filtered

    filtered_dist = compute_topic_distribution(filtered)
    below_threshold: Dict[str, Dict[str, float]] = {}

    for topic, orig_count in original_dist.items():
        if orig_count == 0:
            continue
        filt_count = filtered_dist.get(topic, 0)
        coverage = filt_count / orig_count
        if coverage >= min_coverage:
            continue

        below_threshold[topic] = {
            "original": float(orig_count),
            "filtered": float(filt_count),
            "coverage": round(coverage, 4),
            "allowed_roles": sorted(allowed_roles),
        }

    if below_threshold:
        logger.warning(
            "Cobertura por topic por debajo de %.0f%% tras el filtro accional: %s",
            min_coverage * 100,
            below_threshold,
        )

    return filtered


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # ── Validar API key ──────────────────────────────────────────────────────
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        logger.error(
            "API key no encontrada. Configura la variable OPENAI_API_KEY "
            "o pasa --api-key <key>."
        )
        sys.exit(1)

    # ── Cargar chunks ────────────────────────────────────────────────────────
    if not args.input.exists():
        logger.error("Archivo de entrada no encontrado: %s", args.input)
        sys.exit(1)

    chunks = read_jsonl(args.input)
    logger.info("Chunks cargados: %d desde %s", len(chunks), args.input)

    # ── Filtrar por clinical_score si se pide ────────────────────────────────
    if args.min_clinical_score > 0:
        before = len(chunks)
        chunks = [
            c
            for c in chunks
            if c.get("metadata", {}).get("clinical_score", 0) >= args.min_clinical_score
        ]
        logger.info(
            "Filtro clinical_score >= %d: %d → %d chunks",
            args.min_clinical_score,
            before,
            len(chunks),
        )

    if not chunks:
        logger.warning("Sin chunks para procesar. Verifica --min-clinical-score.")
        sys.exit(0)

    # ── Phase 7: Filtrar por content_role ─────────────────────────────────────
    allowed_roles = set(r.strip() for r in args.allowed_content_roles.split(",") if r.strip())
    filter_stats = {
        "input_chunks": len(chunks),
        "eligible_chunks": len(chunks),
        "skipped_chunks": 0,
        "eligible_by_content_role": {},
        "skipped_by_content_role": {},
        "allowed_content_roles": sorted(allowed_roles),
    }
    if not args.no_content_role_filter:
        original_chunks = chunks[:]
        chunks, filter_stats = filter_by_content_role(chunks, allowed_roles, logger)
        chunks = preserve_topic_coverage(
            original_chunks, chunks, args.min_topic_coverage, allowed_roles, logger
        )
    else:
        logger.info("Filtro por content_role deshabilitado (--no-content-role-filter).")

    # Reportar distribución de topics tras filtro
    topic_dist = compute_topic_distribution(chunks)
    if topic_dist:
        logger.info("Topics tras filtro: %s", dict(sorted(topic_dist.items())))

    if not chunks:
        logger.warning("Sin chunks para procesar tras filtros de content_role.")
        sys.exit(0)

    if args.limit is not None:
        if args.limit <= 0:
            logger.error("--limit debe ser mayor que 0.")
            sys.exit(1)
        before = len(chunks)
        chunks = chunks[: args.limit]
        logger.info("Limitando chunks: %d → %d", before, len(chunks))

    # ── Estimación de costo y estadísticas ──────────────────────────────────
    expected_pairs, cost_est = estimate_cost(
        chunks, args.model, args.min_pairs, args.max_pairs
    )

    print()
    print("=" * 60)
    print("  RESUMEN DE GENERACIÓN SINTÉTICA")
    print("=" * 60)
    print(f"  Modelo             : {args.model}")
    print(f"  Chunks de entrada  : {filter_stats['input_chunks']}")
    print(f"  Chunks a procesar  : {len(chunks)}")
    if not args.no_content_role_filter:
        print(f"  Content roles      : {', '.join(sorted(allowed_roles))}")
        print(f"  Topic coverage min : {args.min_topic_coverage:.0%}")
        print(f"  Chunks excluidos   : {filter_stats['skipped_chunks']}")
        if filter_stats["skipped_by_content_role"]:
            print("  Excluidos por rol  :")
            for role, count in filter_stats["skipped_by_content_role"].items():
                print(f"    - {role}: {count}")
    else:
        print("  Content roles      : (sin filtro)")
    print(f"  Pares esperados    : ~{expected_pairs}")
    print(f"  Costo estimado     : ~${cost_est:.2f} USD")
    print(f"  Concurrencia       : {args.concurrency} peticiones simultáneas")
    print(f"  Salida SFT         : {args.sft_output}")
    print(f"  Salida raw         : {args.raw_output}")
    print(f"  Reporte QA         : {args.report_output}")
    print(f"  Checkpoint         : {args.progress_file}")
    print("=" * 60)
    print()

    if args.dry_run:
        logger.info("Modo dry-run: no se llamó a la API.")
        return

    # ── Ejecutar generación ──────────────────────────────────────────────────
    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.error("openai no está instalado. Ejecuta: pip install 'openai>=1.50.0'")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)

    start = time.monotonic()
    stats = asyncio.run(
        run_generation(
            chunks=chunks,
            client=client,
            model=args.model,
            min_pairs=args.min_pairs,
            max_pairs=args.max_pairs,
            concurrency=args.concurrency,
            sft_output=args.sft_output,
            raw_output=args.raw_output,
            progress_file=args.progress_file,
            report_output=args.report_output,
            logger=logger,
        )
    )
    elapsed = time.monotonic() - start

    print()
    print("=" * 60)
    print("  RESULTADOS")
    print("=" * 60)
    print(f"  Chunks totales     : {stats['total']}")
    print(f"  Ya procesados      : {stats['skipped']}")
    print(f"  Procesados ahora   : {stats['processed']}")
    print(f"  Fallidos           : {stats['failed']}")
    print(f"  Pares QA generados : {stats['qa_pairs']}")
    print(
        "  Grounding (overlap): "
        f"{stats['grounding']['avg_context_answer_overlap']:.3f} "
        f"(bajo={stats['grounding']['low_grounding_rate']:.1%})"
    )
    print(f"  Tiempo             : {elapsed:.0f}s")
    print(f"  SFT output         : {args.sft_output}")
    print(f"  Raw output         : {args.raw_output}")
    print(f"  QA report          : {args.report_output}")
    print("=" * 60)

    if stats["failed"] > 0:
        logger.warning(
            "%d chunks fallaron. Puedes relanzar el script para reintentarlos "
            "(el checkpoint excluirá los ya completados).",
            stats["failed"],
        )


if __name__ == "__main__":
    main()
