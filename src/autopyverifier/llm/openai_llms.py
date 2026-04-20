from __future__ import annotations

import json
import os
import re
from typing import Any, Sequence

from openai import OpenAI

from .base import LLMClient


class OpenAICompatibleClient(LLMClient):
    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY for OpenAI-compatible client")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
    ) -> str:
        data = self._responses_request(
            model=model,
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return data.output_text.strip()

    def build_context(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
    ) -> Any:
        data = self._responses_request(
            model=model,
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        text = data.output_text.strip()
        return _extract_json_object(text)

    def _responses_request(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, str]],
        temperature: float,
        max_output_tokens: int,
    ) -> Any:
        payload: dict[str, Any] = {
            "model": model,
            "input": list(messages),
        }
        # Some models/endpoints may ignore temperature; keep optional and low-risk.
        if temperature is not None:
            payload["temperature"] = temperature

        try:
            return self.client.responses.create(**payload)
        except Exception as e:
            raise RuntimeError(f"OpenAI request failed: {e}") from e


def _extract_json_object(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        return json.loads(candidate)
    raise ValueError("Could not parse JSON context from LLM response")
