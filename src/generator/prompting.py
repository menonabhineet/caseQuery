from typing import List
from textwrap import dedent

from src.retriever.faiss_retriever import RetrievedChunk


SYSTEM_PROMPT = dedent(
    """
    You are a careful legal document assistant.
    You answer questions ONLY using the provided context passages, which are
    extracts from legal contracts and case law.

    Rules:
    - If the answer is not clearly supported by the context, say you do not know.
    - Do NOT invent legal facts or hallucinate citations.
    - When you answer, always mention which passages you relied on, by their IDs.
    - Explain in plain English; you are not giving formal legal advice.
    """
).strip()


def build_user_prompt(question: str, passages: List[RetrievedChunk]) -> str:
    parts = []

    parts.append(f"Question:\n{question}\n")

    parts.append("Context passages:")
    for i, p in enumerate(passages, start=1):
        parts.append(
            f"[{i}] (id={p.chunk_id}, source={p.source}, title={p.title})\n"
            f"{p.text}\n"
        )

    parts.append(
        dedent(
            """
            Instructions:
            - Use ONLY the information contained in the context passages above.
            - If the context does not contain enough information, say:
              "I don't know based on the provided documents."
            - After your answer, list the passages you used, e.g.:
              Sources: [1], [3]
            """
        ).strip()
    )

    return "\n\n".join(parts)


__all__ = ["SYSTEM_PROMPT", "build_user_prompt"]
