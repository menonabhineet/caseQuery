import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MAX_CHUNKS = 480_000

from src.config import (
    CHUNKS_JSONL,
    EMBEDDINGS_NPY,
    CHUNKS_META_JSONL,
    FAISS_INDEX_PATH,
    EMBEDDING_MODEL_NAME,
)


def load_chunks(path: Path, max_chunks: int | None = None) -> List[Dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_chunks is not None and i >= max_chunks:
                break
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def save_meta(chunks: List[Dict], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for rec in chunks:
            meta = {
                "chunk_id": rec["chunk_id"],
                "doc_id": rec["doc_id"],
                "source": rec["source"],
                "title": rec.get("title", ""),
            }
            f.write(json.dumps(meta) + "\n")


def build_embeddings_and_index(
    chunks_path: Path = CHUNKS_JSONL,
    embeddings_path: Path = EMBEDDINGS_NPY,
    meta_path: Path = CHUNKS_META_JSONL,
    index_path: Path = FAISS_INDEX_PATH,
    batch_size: int = 64,
) -> None:
    print("Loading chunks...")
    chunks = load_chunks(chunks_path, max_chunks=MAX_CHUNKS)
    texts = [c["text"] for c in chunks]

    print(f"Loaded {len(texts)} chunks.")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda")

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
        batch_texts = texts[i : i + batch_size]
        emb = model.encode(
            batch_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # for cosine similarity
        )
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings).astype("float32")
    print("Embeddings shape:", embeddings.shape)

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path}")

    # Save metadata
    save_meta(chunks, meta_path)
    print(f"Saved chunk metadata to {meta_path}")

    # Build FAISS index (cosine similarity via inner product on normalized vectors)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index with {index.ntotal} vectors to {index_path}")


if __name__ == "__main__":
    build_embeddings_and_index()
