from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Sequence

from anthropic import Anthropic

from .base import LLMClient


class ClaudeCompatibleClient(LLMClient):
    def __init__(self):
        self.max_retries = 6
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str,
        temperature: None,
        max_output_tokens: None,
    ) -> str:
        system, claude_messages = _split_system_message(messages)
        text = self._responses_request(
            model=model,
            system=system,
            messages=claude_messages,
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
        system, claude_messages = _split_system_message(messages)
        text = self._responses_request(
            model=model,
            system=system,
            messages=claude_messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return _extract_json_object(text)

    def _responses_request(
        self,
        *,
        model: str,
        system: str | None,
        messages: Sequence[dict[str, str]],
        temperature: float | None,
        max_output_tokens: int | None,
    ) -> str:

        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": model,
                    "messages": list(messages),
                    "max_tokens": max_output_tokens or 10000
                } 
                if system:
                    kwargs["system"] = system
                if temperature is not None:
                    kwargs["temperature"] = temperature

                resp = self.client.messages.create(**kwargs)

                parts = []
                for block in resp.content:
                    if getattr(block, "type", None) == "text":
                        parts.append(block.text)
                return "".join(parts)

            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 10))

        raise RuntimeError(f"Claude call failed after {self.max_retries} retries") from last_err


def _split_system_message(
    messages: Sequence[dict[str, str]],
) -> tuple[str | None, list[dict[str, str]]]:
    system_parts = []
    new_messages = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if role == "system":
            system_parts.append(content)
        else:
            new_messages.append({"role": role, "content": content})

    system = "\n\n".join(system_parts) if system_parts else None
    return system, new_messages


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