import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

from rouge_score import rouge_scorer
from bert_score import score as bert_score

from src.config import DATA_DIR
from src.generator.rag_pipeline import LegalRAG
from src.baselines.zero_shot_llm import ZeroShotLegalQA


GEN_EVAL_FILE = DATA_DIR / "generation_eval.jsonl"
RESULTS_FILE = DATA_DIR / "generation_eval_results.jsonl"


@dataclass
class GenExample:
    query: str
    reference_answer: str


def load_examples(path: Path = GEN_EVAL_FILE) -> List[GenExample]:
    examples: List[GenExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            examples.append(
                GenExample(
                    query=obj["query"],
                    reference_answer=obj["reference_answer"],
                )
            )
    return examples


def compute_rouge_l(references: List[str], candidates: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []

    for ref, cand in zip(references, candidates):
        s = scorer.score(ref, cand)["rougeL"]
        scores.append(s.fmeasure)

    if not scores:
        return {"rougeL_f": 0.0}

    return {"rougeL_f": sum(scores) / len(scores)}


def compute_bert_score(references: List[str], candidates: List[str]) -> Dict[str, float]:
    if not references:
        return {"bert_f1": 0.0}

    P, R, F1 = bert_score(
        cands=candidates,
        refs=references,
        lang="en",
        rescale_with_baseline=False,
    )
    return {"bert_f1": float(F1.mean().item())}


def main():
    examples = load_examples()
    print(f"Loaded {len(examples)} generation eval examples.")

    rag = LegalRAG(top_k=5)
    zero_shot = ZeroShotLegalQA()

    # Store outputs for later inspection
    results: List[Dict[str, Any]] = []

    rag_candidates: List[str] = []
    zs_candidates: List[str] = []
    references: List[str] = []

    for ex in examples:
        print(f"\n=== Query: {ex.query} ===")

        # Defaults in case of failure
        rag_text = ""
        zs_text = ""
        # RAG answer
        try:
            rag_ans = rag.answer(ex.query)
            rag_text = rag_ans.answer
            print("RAG answer:")
            print(rag_text[:300].replace("\n", " ") + "..." if len(rag_text) > 300 else rag_text)
        except Exception as e:
            print(f"[RAG ERROR] {e}")

       # Zero-shot
        try:
            zs_ans = zero_shot.answer(ex.query)
            zs_text = zs_ans.answer
            print("\nZero-shot answer:")
            print(zs_text[:300].replace("\n", " ") + "..." if len(zs_text) > 300 else zs_text)
        except Exception as e:
            print(f"[ZERO-SHOT ERROR] {e}")

        rag_candidates.append(rag_text)
        zs_candidates.append(zs_text)
        references.append(ex.reference_answer)

        results.append(
            {
                "query": ex.query,
                "reference_answer": ex.reference_answer,
                "rag_answer": rag_ans.answer,
                "rag_sources": [
                    {
                        "chunk_id": p.chunk_id,
                        "doc_id": p.doc_id,
                        "source": p.source,
                        "title": p.title,
                        "score": p.score,
                    }
                    for p in rag_ans.passages
                ],
                "zero_shot_answer": zs_ans.answer,
            }
        )

    # Save detailed outputs
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved detailed results to {RESULTS_FILE}")

    # Compute metrics for RAG
    rag_rouge = compute_rouge_l(references, rag_candidates)
    rag_bert = compute_bert_score(references, rag_candidates)

    # Compute metrics for zero-shot
    zs_rouge = compute_rouge_l(references, zs_candidates)
    zs_bert = compute_bert_score(references, zs_candidates)

    print("\n=== Aggregated metrics ===")
    print("RAG:")
    print({"rougeL_f": rag_rouge["rougeL_f"], "bert_f1": rag_bert["bert_f1"]})
    print("\nZero-shot:")
    print({"rougeL_f": zs_rouge["rougeL_f"], "bert_f1": zs_bert["bert_f1"]})


if __name__ == "__main__":
    main()
