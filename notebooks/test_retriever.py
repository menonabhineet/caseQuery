from src.retriever.faiss_retriever import FaissRetriever

if __name__ == "__main__":
    retriever = FaissRetriever()
    query = "What is the termination clause in this agreement?"
    results = retriever.search(query, k=5)

    for r in results:
        print(f"{r.score:.3f} | {r.source} | {r.doc_id} | {r.title}")
        print("TEXT SNIPPET:", r.text[:300].replace("\n", " "), "\n")

