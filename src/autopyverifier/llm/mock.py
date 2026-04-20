from __future__ import annotations

import json
import re
from typing import Any, List, Sequence

from .base import LLMClient


MOCK_SEED_RESPONSE = r'''
```python
VERIFIER_SPECS = [
    {
        "name": "has_normalized_text",
        "description": "Checks that the output can be normalized into non-empty text.",
        "requires": ["normalized_text"]
    },
    {
        "name": "not_refusal_like",
        "description": "Rejects obvious refusal-style outputs.",
        "requires": ["normalized_text"]
    }
]


def has_normalized_text(x, y, context=None):
    text = (context or {}).get("normalized_text")
    return isinstance(text, str) and len(text.strip()) > 0


def not_refusal_like(x, y, context=None):
    text = ((context or {}).get("normalized_text") or "").lower()
    banned = ["i can't", "i cannot", "unable to", "don't know"]
    return not any(token in text for token in banned)


def aggregate(checks, x, y, context=None):
    return all(checks.values())
```

```python
VERIFIER_SPECS = [
    {
        "name": "has_normalized_text",
        "description": "Checks that normalized text exists.",
        "requires": ["normalized_text"]
    },
    {
        "name": "has_final_answer_signal",
        "description": "Checks that a final answer can be identified or inferred.",
        "requires": ["final_answer", "normalized_text"]
    }
]


def has_normalized_text(x, y, context=None):
    text = (context or {}).get("normalized_text")
    return isinstance(text, str) and len(text.strip()) > 0


def has_final_answer_signal(x, y, context=None):
    ctx = context or {}
    answer = ctx.get("final_answer")
    text = ctx.get("normalized_text") or ""
    return answer is not None or len(text.split()) >= 3


def aggregate(checks, x, y, context=None):
    return all(checks.values())
```
'''.strip()


MOCK_CRITIC_RESPONSE = """
1. Main loopholes causing false positives
- Surface checks accept outputs that look well-formed but may still be wrong.

2. Main over-restrictions causing false negatives
- Bundles may reject valid outputs that phrase the answer differently.

3. Redundant or weak checks
- Repeated non-empty checks add little value.

4. Missing verifier types or missing context fields
- Consider extracting a stronger final_answer field.
- Consider a lightweight consistency check over normalized_text and final_answer.

5. Suggested structured edits (ADD / REMOVE / REPLACE / MODIFY / CHANGE_AGGREGATOR)
- ADD one final-answer oriented verifier.
- REPLACE weak format-only checks when possible.
- Keep the set small.
""".strip()


MOCK_REFINE_RESPONSE = r'''
```python
VERIFIER_SPECS = [
    {
        "name": "has_normalized_text",
        "description": "Checks that normalized text exists.",
        "requires": ["normalized_text"]
    },
    {
        "name": "not_refusal_like",
        "description": "Rejects obvious refusal-style outputs.",
        "requires": ["normalized_text"]
    },
    {
        "name": "has_final_answer_signal",
        "description": "Checks that a final answer can be identified or inferred.",
        "requires": ["final_answer", "normalized_text"]
    }
]


def has_normalized_text(x, y, context=None):
    text = (context or {}).get("normalized_text")
    return isinstance(text, str) and len(text.strip()) > 0


def not_refusal_like(x, y, context=None):
    text = ((context or {}).get("normalized_text") or "").lower()
    banned = ["i can't", "i cannot", "unable to", "don't know"]
    return not any(token in text for token in banned)


def has_final_answer_signal(x, y, context=None):
    ctx = context or {}
    answer = ctx.get("final_answer")
    text = ctx.get("normalized_text") or ""
    return answer is not None or len(text.split()) >= 3


def aggregate(checks, x, y, context=None):
    return all(checks.values())
```

```python
VERIFIER_SPECS = [
    {
        "name": "has_normalized_text",
        "description": "Checks that normalized text exists.",
        "requires": ["normalized_text"]
    },
    {
        "name": "has_numbers_when_relevant",
        "description": "Requires at least one extracted number when the output appears numeric.",
        "requires": ["normalized_text", "numbers"]
    }
]


def has_normalized_text(x, y, context=None):
    text = (context or {}).get("normalized_text")
    return isinstance(text, str) and len(text.strip()) > 0


def has_numbers_when_relevant(x, y, context=None):
    ctx = context or {}
    text = (ctx.get("normalized_text") or "").lower()
    nums = ctx.get("numbers") or []
    if any(token in text for token in ["x=", "answer", "result", "root", "roots"]):
        return len(nums) >= 1
    return True


def aggregate(checks, x, y, context=None):
    return all(checks.values())
```
'''.strip()


def _normalize_text(y: Any) -> str:
    if isinstance(y, str):
        return y
    if isinstance(y, dict):
        try:
            return json.dumps(y, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(y)
    return str(y)


def _extract_final_answer(text: str):
    patterns = [
        r"answer\s+is\s+([^\.\n]+)",
        r"hence\s+([^\.\n]+)",
        r"therefore\s+([^\.\n]+)",
        r"so\s+([^\.\n]+)",
        r"roots?\s+are\s+([^\.\n]+)",
        r"x\s*=\s*([^\.\n]+)",
    ]
    lower = text.lower()
    for pat in patterns:
        m = re.search(pat, lower)
        if m:
            return m.group(1).strip()
    return None


def _extract_numbers(text: str):
    return re.findall(r"[-+]?\d+(?:\.\d+)?", text)


class MockLLMClient(LLMClient):
    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
    ) -> str:
        model_l = model.lower()
        if "critic" in model_l:
            return MOCK_CRITIC_RESPONSE
        if "refine" in model_l:
            return MOCK_REFINE_RESPONSE
        return MOCK_SEED_RESPONSE

    def build_context(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
    ) -> Any:
        prompt = "\n".join(m.get("content", "") for m in messages)
        # Very lightweight parsing of required fields and y from the prompt.
        req_match = re.search(r"Required fields:\n(\[.*?\])", prompt, flags=re.DOTALL)
        required_fields = []
        if req_match:
            try:
                required_fields = json.loads(req_match.group(1))
            except Exception:
                required_fields = []

        y_match = re.search(r"Model output y:\n(.*?)(?:\n\nRequired fields:|$)", prompt, flags=re.DOTALL)
        y_raw = None
        if y_match:
            y_txt = y_match.group(1).strip()
            try:
                y_raw = json.loads(y_txt)
            except Exception:
                y_raw = y_txt

        text = _normalize_text(y_raw)
        context = {}
        for field in required_fields:
            if field == "raw_output":
                context[field] = y_raw
            elif field == "normalized_text":
                context[field] = text
            elif field == "final_answer":
                context[field] = _extract_final_answer(text)
            elif field == "numbers":
                context[field] = _extract_numbers(text)
            elif field == "lines":
                context[field] = [ln for ln in text.splitlines() if ln.strip()]
            elif field == "json_obj":
                context[field] = y_raw if isinstance(y_raw, (dict, list)) else None
            else:
                context[field] = None
        return context
