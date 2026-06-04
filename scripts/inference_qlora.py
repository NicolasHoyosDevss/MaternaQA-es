#!/usr/bin/env python3
"""Run inference on QLoRA adapters using the same 4-bit base model as training.

Loads the base model with 4-bit BitsAndBytes quantization, mounts the PEFT
adapter, and generates completions for the test split in prompt/completion
format — matching how train_qlora_trl.py trained.

Usage:
    python scripts/inference_qlora.py \
        --adapter-dir outputs/gemma4-grounded \
        --output-prefix outputs/gemma4-grounded/test

    python scripts/inference_qlora.py \
        --adapter-dir outputs/medgemma-grounded \
        --output-prefix outputs/medgemma-grounded/test \
        --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


DATASET_ROOT = Path("datasets/obstetrics/qa/publication")


def remove_project_root_from_imports() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path = [
        entry
        for entry in sys.path
        if entry and Path(entry).resolve() != project_root
    ]


def import_stack() -> dict[str, Any]:
    remove_project_root_from_imports()
    try:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        raise SystemExit(
            "Faltan dependencias. Instala requirements.txt primero. "
            f"Error: {exc}"
        ) from exc
    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForImageTextToText": AutoModelForImageTextToText,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferencia con adapters QLoRA entrenados por train_qlora_trl.py"
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        required=True,
        help="Directorio con adapter_model.safetensors y adapter_config.json",
    )
    parser.add_argument(
        "--dataset-variant",
        default="sft_grounded",
        choices=("sft_closed_book", "sft_grounded"),
        help="Variante SFT a evaluar.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Prefijo para archivos de salida (se genera <prefix>_predictions.jsonl)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9,
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        default=False,
        help="Usa muestreo estocastico. Por defecto se usa greedy decoding reproducible para evaluacion.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cantidad maxima de ejemplos a evaluar (para smoke tests).",
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true", default=False,
    )
    parser.add_argument(
        "--attn-implementation", default=None,
    )
    return parser.parse_args()


def load_adapter_config(adapter_dir: Path) -> dict[str, Any]:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontro adapter_config.json en {adapter_dir}")
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def load_model_and_tokenizer(
    adapter_dir: Path, args: argparse.Namespace, stack: dict[str, Any]
) -> tuple[Any, Any, str]:
    adapter_config = load_adapter_config(adapter_dir)
    base_model = adapter_config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError("adapter_config.json no contiene base_model_name_or_path")

    torch = stack["torch"]
    if not torch.cuda.is_available():
        raise SystemExit("Se requiere GPU NVIDIA con CUDA para inferencia 4-bit.")

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )

    bnb_config = stack["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = stack["AutoTokenizer"].from_pretrained(
        base_model,
        trust_remote_code=args.trust_remote_code,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": compute_dtype,
        "token": os.environ.get("HF_TOKEN"),
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    normalized = base_model.lower()
    if "gemma-4" in normalized or "medgemma" in normalized:
        base = stack["AutoModelForImageTextToText"].from_pretrained(
            base_model, **model_kwargs
        )
    else:
        base = stack["AutoModelForCausalLM"].from_pretrained(
            base_model, **model_kwargs
        )

    model = stack["PeftModel"].from_pretrained(base, str(adapter_dir))
    model.eval()

    print(f"Modelo base: {base_model}")
    print(f"Adapter cargado desde: {adapter_dir}")
    return model, tokenizer, base_model


def build_prompt(
    messages: list[dict[str, str]], tokenizer: Any
) -> str:
    assistant_indices = [
        i
        for i, m in enumerate(messages)
        if isinstance(m, dict) and m.get("role") == "assistant"
    ]
    if not assistant_indices:
        raise ValueError("No se encontro mensaje assistant en el ejemplo.")

    completion_start = assistant_indices[-1]
    prompt_messages = messages[:completion_start]

    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_reference_answer(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "").strip()
    return ""


def extract_question(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = str(message.get("content") or "").strip()
        marker = "Pregunta:"
        if marker in content:
            return content.split(marker, 1)[1].strip()
        return content
    return ""


def load_test_examples(
    dataset_variant: str, limit: int | None = None
) -> list[dict[str, Any]]:
    test_file = DATASET_ROOT / dataset_variant / "test.jsonl"
    if not test_file.exists():
        raise FileNotFoundError(f"No se encontro test.jsonl en {test_file}")

    examples = []
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if limit:
        examples = examples[:limit]
    return examples


def run_inference(
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
    torch: Any,
) -> list[dict[str, Any]]:
    results = []
    total = len(examples)
    start_time = time.time()

    for idx, example in enumerate(examples):
        messages = example.get("messages", [])
        prompt_text = build_prompt(messages, tokenizer)

        inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.do_sample:
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = args.top_p

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                **generation_kwargs,
            )

        generated = tokenizer.decode(
            generated_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        results.append(
            {
                "id": example.get("metadata", {}).get("qa_id", idx),
                "question": extract_question(messages),
                "generated_answer": generated.strip(),
                "reference_answer": extract_reference_answer(messages),
                "source_context": example.get("metadata", {}).get("contexto_fuente", ""),
                "prompt_text": prompt_text,
                "generated": generated.strip(),
                "reference_messages": messages,
                "metadata": example.get("metadata", {}),
            }
        )

        if (idx + 1) % 10 == 0 or idx == total - 1:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{idx + 1}/{total}] {rate:.1f} ejemplos/seg")

    print(f"Completado en {time.time() - start_time:.1f}s")
    return results


def save_results(results: list[dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8", newline="\n") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Resultados guardados en: {output_file}")


def main() -> None:
    args = parse_args()
    stack = import_stack()

    model, tokenizer, base_model = load_model_and_tokenizer(
        args.adapter_dir, args, stack
    )
    examples = load_test_examples(args.dataset_variant, args.limit)
    print(f"Ejemplos cargados: {len(examples)}")

    results = run_inference(model, tokenizer, examples, args, stack["torch"])
    output_file = Path(str(args.output_prefix) + "_predictions.jsonl")
    save_results(results, output_file)


if __name__ == "__main__":
    main()
