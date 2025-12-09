import json
from pathlib import Path
from typing import Dict, Iterator, List

from src.config import CORPUS_JSONL, CHUNKS_JSONL


def split_into_chunks(
    text: str,
    chunk_size: int = 512,
    overlap: int = 128,
) -> List[str]:
    """
    Simple word-based chunking.
    chunk_size & overlap are in *words* (approximation of tokens).
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    n = len(words)

    while start < n:
        end = min(start + chunk_size, n)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == n:
            break
        start = end - overlap  # step back for overlap

    return chunks


def iter_corpus_docs(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_chunks(
    corpus_path: Path = CORPUS_JSONL,
    out_path: Path = CHUNKS_JSONL,
    chunk_size: int = 512,
    overlap: int = 128,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_docs = 0
    total_chunks = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for doc in iter_corpus_docs(corpus_path):
            total_docs += 1
            doc_id = doc["doc_id"]
            source = doc["source"]
            title = doc.get("title", "")

            chunks = split_into_chunks(doc["text"], chunk_size, overlap)
            for idx, chunk_text in enumerate(chunks):
                chunk_id = f"{doc_id}::chunk{idx}"
                record = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "source": source,
                    "title": title,
                    "text": chunk_text,
                }
                out_f.write(json.dumps(record) + "\n")
                total_chunks += 1

    print(
        f"Chunked {total_docs} documents into {total_chunks} chunks. "
        f"Saved to {out_path}"
    )


if __name__ == "__main__":
    build_chunks()
