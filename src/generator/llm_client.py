import os
from abc import ABC, abstractmethod
from typing import Optional

import requests


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        ...


class OpenRouterChatClient(BaseLLMClient):
    """
    Thin wrapper around OpenRouter's /chat/completions endpoint.

    Requires:
      - OPENROUTER_API_KEY in environment.
    Optional:
      - OPENROUTER_SITE
      - OPENROUTER_TITLE
    """

    def __init__(
        self,
        model: str = "meta-llama/llama-3.3-70b-instruct:free",
        temperature: float = 0.1,
    ):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set in environment.")

        self.api_key = api_key
        self.model = model
        self.temperature = temperature

        # Optional headers OpenRouter recommends
        self.site = os.getenv("OPENROUTER_SITE", "https://example.com")
        self.title = os.getenv("OPENROUTER_TITLE", "CS582 CaseQuery")

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site,
            "X-Title": self.title,
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }

        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()

        data = resp.json()
        # Standard Chat Completions-style structure
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"Unexpected OpenRouter response format: {data}")
