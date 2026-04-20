from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .models import Example


def load_devset(path: str | Path) -> List[Example]:
    examples: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            examples.append(
                Example(
                    id=str(obj.get("id", idx)),
                    x=obj["query"],
                    y=obj["output"],
                    objective=int(obj["objective"]),
                    metadata=obj.get("metadata", {}),
                )
            )
    return examples


def format_examples_for_prompt(examples: Iterable[Example], include_labels: bool = True) -> str:
    rows = []
    for ex in examples:
        row = {"Query": ex.x, "Output": ex.y}
        if include_labels:
            row["objective"] = ex.objective
        rows.append(row)
    return json.dumps(rows, ensure_ascii=False, indent=2)
