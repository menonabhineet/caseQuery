import argparse
import sys
from textwrap import fill

from src.generator.rag_pipeline import LegalRAG
from src.baselines.zero_shot_llm import ZeroShotLegalQA


def print_header(mode: str):
    print("=" * 80)
    print(f"CaseQuery CLI – mode: {mode.upper()}")
    print("Type your legal question and press Enter.")
    print("Type 'exit' or 'quit' to leave.")
    print("=" * 80)


def wrap(text: str, width: int = 80) -> str:
    return fill(text, width=width)


def run_rag():
    rag = LegalRAG(top_k=5)
    print_header("rag")

    while True:
        try:
            question = input("\nQ> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Goodbye!")
            break

        if question.lower() in {"exit", "quit"}:
            print("Exiting. Goodbye!")
            break

        if not question:
            continue

        print("\nRetrieving and generating answer...\n")
        result = rag.answer(question)

        print("ANSWER:\n")
        print(wrap(result.answer))
        print("\nSOURCES:\n")
        for i, p in enumerate(result.passages, start=1):
            meta_line = f"[{i}] {p.source} | {p.doc_id} | {p.title} | score={p.score:.3f}"
            print(wrap(meta_line))


def run_zero_shot():
    zs = ZeroShotLegalQA()
    print_header("zero-shot")

    while True:
        try:
            question = input("\nQ> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Goodbye!")
            break

        if question.lower() in {"exit", "quit"}:
            print("Exiting. Goodbye!")
            break

        if not question:
            continue

        print("\nGenerating answer (no retrieval)...\n")
        result = zs.answer(question)

        print("ANSWER:\n")
        print(wrap(result.answer))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="CaseQuery CLI demo: RAG vs zero-shot legal QA"
    )
    parser.add_argument(
        "--mode",
        choices=["rag", "zero-shot"],
        default="rag",
        help="Which mode to run: 'rag' (retrieval-augmented) or 'zero-shot'",
    )

    args = parser.parse_args(argv)

    if args.mode == "rag":
        run_rag()
    else:
        run_zero_shot()


if __name__ == "__main__":
    main()
