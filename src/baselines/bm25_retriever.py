import json
from pathlib import Path
from typing import List, Optional

from rank_bm25 import BM25Okapi

from src.config import CHUNKS_JSONL
from src.retriever.faiss_retriever import RetrievedChunk


def _tokenize(text: str) -> List[str]:
    # Very simple tokenizer: lowercase + split on whitespace
    return text.lower().split()


class BM25Retriever:
    """
    BM25 baseline retriever using rank_bm25 on the chunk corpus.

    NOTE: Building the index may take a bit of time at startup
    because we load and tokenize all chunks.
    """

    def __init__(self, chunks_path: Path = CHUNKS_JSONL, max_docs: Optional[int] = 50000):
        """
        max_docs: if not None, only the first max_docs chunks are used.
        This keeps memory and build time under control.
        """
        self.chunks_path = chunks_path
        self.max_docs = max_docs
        self.chunks_meta: List[dict] = []
        self.chunks_texts: List[str] = []
        self.tokenized_corpus: List[List[str]] = []

        self._load_chunks()
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(
            f"BM25Retriever initialized with {len(self.chunks_meta)} chunks "
            f"from {self.chunks_path}"
        )

    def _load_chunks(self) -> None:
        with self.chunks_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if self.max_docs is not None and i >= self.max_docs:
                    break

                rec = json.loads(line.strip())
                self.chunks_meta.append(
                    {
                        "chunk_id": rec["chunk_id"],
                        "doc_id": rec["doc_id"],
                        "source": rec["source"],
                        "title": rec.get("title", ""),
                    }
                )
                text = rec["text"]
                self.chunks_texts.append(text)
                self.tokenized_corpus.append(_tokenize(text))

    def search(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get indices of top-k scores
        if k >= len(scores):
            top_idx = list(range(len(scores)))
        else:
            top_idx = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:k]

        results: List[RetrievedChunk] = []
        for idx in top_idx:
            m = self.chunks_meta[idx]
            text = self.chunks_texts[idx]
            score = float(scores[idx])

            results.append(
                RetrievedChunk(
                    chunk_id=m["chunk_id"],
                    doc_id=m["doc_id"],
                    source=m["source"],
                    title=m.get("title", ""),
                    text=text,
                    score=score,
                )
            )

        return results
