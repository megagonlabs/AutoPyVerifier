from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
import time
from typing import Any, Sequence
from google import genai
from google.genai import types

from .base import LLMClient


class GeminiCompatibleClient(LLMClient):
    def __init__(self):
        self.max_retries = 6
        self.client = genai.Client()


    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str,
        temperature: None,
        max_output_tokens: None,
    ) -> str:
        prompt = _messages_to_prompt(messages)
        text = self._responses_request(
            model=model,
            input_text=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return text

    def build_context(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str,
        temperature: None,
        max_output_tokens: None,
    ) -> Any:
        prompt = _messages_to_prompt(messages)
        text = self._responses_request(
            model=model,
            input_text=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return _extract_json_object(text)


    def _responses_request(
        self,
        *,
        model: str,
        input_text: str,
        temperature: float | None,
        max_output_tokens: int | None,
    ) -> dict:


        for attempt in range(self.max_retries):
            try:
                cfg = types.GenerateContentConfig()

                if temperature is not None:
                    cfg.temperature = temperature
                if max_output_tokens is not None:
                    cfg.max_output_tokens = max_output_tokens
                resp = self.client.models.generate_content(
                    model=model,
                    contents=input_text,
                    config=cfg,
                )

                return resp.text 
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 10))

        raise RuntimeError(f"Gemini call failed after {self.max_retries} retries") from last_err


def _messages_to_prompt(messages: Sequence[dict[str, str]]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


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
