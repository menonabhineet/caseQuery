from dataclasses import dataclass
from typing import List

from src.retriever.faiss_retriever import FaissRetriever, RetrievedChunk
from src.generator.llm_client import BaseLLMClient, OpenRouterChatClient
from src.generator.prompting import SYSTEM_PROMPT, build_user_prompt


@dataclass
class RAGAnswer:
    question: str
    answer: str
    passages: List[RetrievedChunk]


class LegalRAG:
    def __init__(
        self,
        retriever: FaissRetriever | None = None,
        llm_client: BaseLLMClient | None = None,
        top_k: int = 5,
    ):
        self.retriever = retriever or FaissRetriever()
        self.llm = llm_client or OpenRouterChatClient()
        self.top_k = top_k

    def answer(self, question: str, top_k: int | None = None, max_tokens: int = 512) -> RAGAnswer:
        k = top_k or self.top_k

        # 1. Retrieve passages
        passages = self.retriever.search(question, k=k)

        # 2. Build prompt
        user_prompt = build_user_prompt(question, passages)

        # 3. Call LLM via OpenRouter
        answer = self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )

        return RAGAnswer(
            question=question,
            answer=answer,
            passages=passages,
        )
