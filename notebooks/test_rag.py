from src.generator.rag_pipeline import LegalRAG


if __name__ == "__main__":
    rag = LegalRAG(top_k=5)

    question = "What are the termination conditions in these agreements?"
    result = rag.answer(question)

    print("QUESTION:\n", result.question)
    print("\nANSWER:\n", result.answer)
    print("\nSOURCES:")
    for i, p in enumerate(result.passages, start=1):
        print(f"[{i}] {p.source} | {p.doc_id} | {p.title} | score={p.score:.3f}")
