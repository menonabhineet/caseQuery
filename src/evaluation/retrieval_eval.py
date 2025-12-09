import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

from src.config import DATA_DIR
from src.retriever.faiss_retriever import FaissRetriever
from src.baselines.bm25_retriever import BM25Retriever


EVAL_FILE = DATA_DIR / "eval_queries.jsonl"


@dataclass
class EvalExample:
    query: str
    relevant_chunk_ids: List[str]


def load_eval_examples(path: Path = EVAL_FILE) -> List[EvalExample]:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {path}\n"
            f"Create it with one JSON object per line:\n"
            f'  {{"query": "...", "relevant_chunk_ids": ["chunk_id1", "chunk_id2"]}}'
        )

    examples: List[EvalExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            examples.append(
                EvalExample(
                    query=obj["query"],
                    relevant_chunk_ids=obj["relevant_chunk_ids"],
                )
            )
    return examples


def compute_metrics(
    ranks: List[int],
    k: int,
) -> Dict[str, float]:
    """
    ranks: list of 1-based ranks of a relevant doc for each query,
           or 0 if no relevant doc retrieved in top-k.
    """
    n = len(ranks)
    if n == 0:
        return {"recall@k": 0.0, "mrr@k": 0.0}

    recall = sum(1 for r in ranks if r > 0 and r <= k) / n
    mrr = sum(1.0 / r for r in ranks if r > 0 and r <= k) / n
    return {"recall@k": recall, "mrr@k": mrr}


def evaluate_retriever(
    examples: List[EvalExample],
    retriever,
    k: int = 10,
) -> Dict[str, float]:
    ranks: List[int] = []

    for ex in examples:
        results = retriever.search(ex.query, k=k)

        # Map chunk_id -> rank (1-based)
        rank_map: Dict[str, int] = {}
        for idx, r in enumerate(results, start=1):
            rank_map[r.chunk_id] = idx

        # Best rank among any relevant chunk_ids
        best_rank = 0
        for rel_id in ex.relevant_chunk_ids:
            if rel_id in rank_map:
                r = rank_map[rel_id]
                if best_rank == 0 or r < best_rank:
                    best_rank = r

        ranks.append(best_rank)

    return compute_metrics(ranks, k)


def main(k: int = 10):
    examples = load_eval_examples()

    print(f"Loaded {len(examples)} evaluation examples.")

    print("\nEvaluating FAISS dense retriever...")
    dense_retriever = FaissRetriever()
    dense_metrics = evaluate_retriever(examples, dense_retriever, k=k)
    print(f"Dense retriever metrics (k={k}): {dense_metrics}")

    print("\nEvaluating BM25 retriever...")
    bm25_retriever = BM25Retriever()
    bm25_metrics = evaluate_retriever(examples, bm25_retriever, k=k)
    print(f"BM25 retriever metrics (k={k}): {bm25_metrics}")


if __name__ == "__main__":
    main(k=10)
