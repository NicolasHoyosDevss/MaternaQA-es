#!/usr/bin/env python3
"""PoC: verify Ragas can accept a LangChain/OpenAI model id (e.g., gpt-5.5)."""

import argparse
import os
import sys
import warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.5", help="OpenAI model id to test")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also do a real API call with LangChain invoke() (requires valid OPENAI_API_KEY and network).",
    )
    parser.add_argument(
        "--allow-dummy-key",
        action="store_true",
        help="If OPENAI_API_KEY is missing, set a dummy key to run constructor-only checks.",
    )
    args = parser.parse_args()

    key = os.getenv("OPENAI_API_KEY")
    if not key and args.allow_dummy_key:
        os.environ["OPENAI_API_KEY"] = "dummy"
        print("[info] OPENAI_API_KEY missing -> using dummy key for constructor-only PoC.")
    elif not key:
        print("[error] OPENAI_API_KEY is missing.")
        print("Set it, or rerun with --allow-dummy-key for a non-live constructor test.")
        return 2

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    print("[step] Build LangChain model objects")
    lc_llm = ChatOpenAI(model=args.model, temperature=0)
    lc_emb = OpenAIEmbeddings(model=args.embedding_model)
    print(f"  - ChatOpenAI(model={args.model!r}) ✅")
    print(f"  - OpenAIEmbeddings(model={args.embedding_model!r}) ✅")

    print("[step] Wrap with Ragas wrappers")
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        ragas_llm = LangchainLLMWrapper(lc_llm)
        ragas_emb = LangchainEmbeddingsWrapper(lc_emb)
    print(f"  - LangchainLLMWrapper -> {type(ragas_llm).__name__} ✅")
    print(f"  - LangchainEmbeddingsWrapper -> {type(ragas_emb).__name__} ✅")

    dep_warnings = [w for w in ws if "deprecated" in str(w.message).lower()]
    if dep_warnings:
        print("[note] Ragas emitted deprecation warnings for LangChain wrappers (expected on recent versions).")
        print("       This PoC still proves model-id compatibility path today.")

    if args.live:
        print("[step] Live ping to OpenAI via LangChain invoke()")
        reply = lc_llm.invoke("Reply ONLY with: OK")
        content = getattr(reply, "content", str(reply))
        print(f"  - Response: {content!r}")
        print("  - Live invocation ✅")
    else:
        print("[skip] Live invocation not requested (--live).")

    print("\nRESULT: Ragas accepted the LangChain model object with model id:", args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
