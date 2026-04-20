from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Sequence


class LLMClient(ABC):
    @abstractmethod
    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str,
        temperature: None,
        max_output_tokens: None,
    ) -> str:
        raise NotImplementedError

    def build_context(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str,
        temperature: None,
        max_output_tokens: None,
    ) -> Any:
        """Return parsed JSON-like context. Backends may override for stronger guarantees."""
        return self.complete(
            messages,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
