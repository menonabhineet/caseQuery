from dataclasses import dataclass
from textwrap import dedent
from typing import Optional

from src.generator.llm_client import BaseLLMClient, OpenRouterChatClient


ZERO_SHOT_SYSTEM_PROMPT = dedent(
    """
    You are a general-purpose legal question answering assistant.
    You will answer legal questions using your own knowledge and reasoning.
    You are NOT given any external context documents.
    Do your best to answer clearly in plain English, but do not claim
    to rely on specific documents or passages.
    You are not giving formal legal advice.
    """
).strip()


@dataclass
class ZeroShotAnswer:
    question: str
    answer: str


class ZeroShotLegalQA:
    """
    Baseline that calls the LLM directly without retrieval.
    Uses the OpenRouterChatClient defined in src/generator/llm_client.py
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        # Default to your OpenRouter-based client
        self.llm: BaseLLMClient = llm_client or OpenRouterChatClient()

    def answer(self, question: str, max_tokens: int = 512) -> ZeroShotAnswer:
        user_prompt = f"Question:\n{question}\n\nAnswer in 2–4 paragraphs."
        resp = self.llm.generate(
            system_prompt=ZERO_SHOT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )
        return ZeroShotAnswer(question=question, answer=resp)
