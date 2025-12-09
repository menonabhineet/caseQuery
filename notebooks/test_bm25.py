from src.baselines.bm25_retriever import BM25Retriever


if __name__ == "__main__":
    # Use a subset for the baseline, e.g. first 50k chunks
    retriever = BM25Retriever(max_docs=50000)

    query = "What are the termination conditions in these agreements?"
    results = retriever.search(query, k=5)

    for r in results:
        print(f"{r.score:.3f} | {r.source} | {r.doc_id} | {r.title}")
