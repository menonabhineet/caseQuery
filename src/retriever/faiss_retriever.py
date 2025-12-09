import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDINGS_NPY,
    CHUNKS_META_JSONL,
    FAISS_INDEX_PATH,
    EMBEDDING_MODEL_NAME,
    CHUNKS_JSONL,  # <-- we’ll add this import below in config.py
)


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    text: str
    score: float


class FaissRetriever:
    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        embeddings_path: Path = EMBEDDINGS_NPY,
        meta_path: Path = CHUNKS_META_JSONL,
        chunks_path: Path = CHUNKS_JSONL,
        model_name: str = EMBEDDING_MODEL_NAME,
    ):
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load embeddings just to get vector dim (optional, but handy)
        self.embeddings = np.load(embeddings_path).astype("float32")
        self.dimension = self.embeddings.shape[1]

        # Load metadata + texts aligned with embedding order
        self.meta, self.texts = self._load_meta_and_text(meta_path, chunks_path)

        # Embedding model for queries
        self.model = SentenceTransformer(model_name)

        print(
            f"Loaded retriever with {len(self.meta)} chunks, "
            f"embedding dim={self.dimension}"
        )

    @staticmethod
    def _load_meta_and_text(
        meta_path: Path,
        chunks_path: Path,
    ) -> (List[Dict], List[str]):
        """
        Assumes meta_path and chunks_path were created from the same
        list in the same order (as in embed_and_index.py).
        """
        meta = []
        texts = []

        with meta_path.open("r", encoding="utf-8") as fm, \
             chunks_path.open("r", encoding="utf-8") as fc:

            for meta_line, chunk_line in zip(fm, fc):
                m = json.loads(meta_line.strip())
                c = json.loads(chunk_line.strip())
                meta.append(m)
                texts.append(c["text"])

        return meta, texts

    def _encode_query(self, query: str) -> np.ndarray:
        vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec.astype("float32")

    def search(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        """
        Return top-k chunks for the query.
        """
        q_vec = self._encode_query(query)
        scores, idxs = self.index.search(q_vec, k)  # shapes: (1, k)
        scores = scores[0]
        idxs = idxs[0]

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores, idxs):
            if idx == -1:
                continue  # no result
            m = self.meta[idx]
            text = self.texts[idx]

            results.append(
                RetrievedChunk(
                    chunk_id=m["chunk_id"],
                    doc_id=m["doc_id"],
                    source=m["source"],
                    title=m.get("title", ""),
                    text=text,
                    score=float(score),
                )
            )
        return results
