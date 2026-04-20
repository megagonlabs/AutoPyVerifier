from __future__ import annotations

import json
from typing import Iterable, List

from .data import format_examples_for_prompt
from .models import Example, Node


VERIFIER_CONTRACT = """
You must output one or more Python modules in fenced ```python code blocks.
Each module must follow this exact contract:

VERIFIER_SPECS = [
    {
        "name": "verifier_name",
        "description": "what it checks",
        "requires": ["field_a", "field_b"]
    },
    ...
]

# one Python function per verifier name
# signature: verifier_name(x, y, context=None)
# each returns True or False
# each verifier should use the context fields declared in VERIFIER_SPECS when possible

def verifier_name(x, y, context=None):
    ...
    return True or False

# final accept/reject rule
# signature: aggregate(checks, x, y, context=None)
# checks is a dict: verifier_name -> bool

def aggregate(checks, x, y, context=None):
    return all(checks.values())

Rules:
- deterministic Python only
- no network access
- no filesystem access
- no subprocesses
- allowed imports only from: math, re, json, statistics, fractions, decimal, itertools, ast, collections
- verifier sets should be small and interpretable
- do not hard-code labels from the dev set
- do not rely on exact gold-string matching
- do not parse raw y repeatedly inside every verifier if a context field can capture the needed information
- prefer generic checks reusable across tasks when possible, but task-specific checks are allowed when justified
- Do not use compile, exec, eval, open, input, globals, locals, vars, getattr, setattr, delattr, __import__, breakpoint, help, or any dynamic code execution/introspection. 
""".strip()


CONTEXT_BUILDER_CONTRACT = """
You build a JSON context for deterministic Python verifiers.
Given a task description, input x, model output y, and a list of required field names, return ONLY a JSON object.

Rules:
- Return exactly one JSON object and nothing else.
- Include every required field exactly once.
- If a field cannot be extracted reliably, set it to null.
- Keep values concise but useful.
- Do not judge whether the output is correct overall.
- Extract structure from y; do not invent unsupported facts.
- Use evidence directly from x and y when possible.
""".strip()


def build_seed_messages(task_description: str, examples: List[Example], num_seeds: int) -> List[dict[str, str]]:
    system = (
        "You are an expert verifier-synthesis assistant. "
        "Your job is to propose a small initial set of deterministic Python verifier bundles "
        "for judging whether a model output satisfies a target objective."
    )
    user = (
        f"Task description:\n{task_description}\n\n"
        f"Need {num_seeds} diverse initial verifier bundles.\n\n"
        f"Development examples (labeled):\n{format_examples_for_prompt(examples, include_labels=True)}\n\n"
        f"{VERIFIER_CONTRACT}\n\n"
        "Generate diverse, small verifier sets that are reusable across similar tasks. "
        "Each verifier must declare a 'requires' list describing which context fields it needs. "
        "Use context fields such as normalized_text, final_answer, steps, equations, json_obj, citations, numbers, entities, or task-specific fields when justified. "
        "Do not use compile, exec, eval, open, input, globals, locals, vars, getattr, setattr, delattr, __import__, breakpoint, help, or any dynamic code execution/introspection. "
        "Do not include any extra explanation. "
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_critic_messages(task_description: str, node: Node, false_pos: List[Example], false_neg: List[Example]) -> List[dict[str, str]]:
    system = (
        "You are a verifier critic. Analyze where the current verifier set fails, "
        "with special attention to false positives, false negatives, redundant checks, and missing context fields."
    )
    user = (
        f"Task description:\n{task_description}\n\n"
        f"Current verifier code:\n```python\n{node.program.source_code}\n```\n\n"
        f"Current metrics: PP={node.stats.pp:.4f}, NP={node.stats.np:.4f}, "
        f"cov_pos={node.stats.cov_pos:.4f}, cov_neg={node.stats.cov_neg:.4f}, size={node.program.size}\n\n"
        f"False positives (accepted but objective=0):\n{format_examples_for_prompt(false_pos, include_labels=True)}\n\n"
        f"False negatives (rejected but objective=1):\n{format_examples_for_prompt(false_neg, include_labels=True)}\n\n"
        "Write a concise diagnosis with these sections:\n"
        "1. Main loopholes causing false positives\n"
        "2. Main over-restrictions causing false negatives\n"
        "3. Redundant or weak checks\n"
        "4. Missing verifier types or missing context fields\n"
        "5. Suggested structured edits (ADD / REMOVE / REPLACE / MODIFY / CHANGE_AGGREGATOR)\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_refine_messages(
    task_description: str,
    node: Node,
    critic_summary: str,
    false_pos: List[Example],
    false_neg: List[Example],
    num_children: int,
) -> List[dict[str, str]]:
    system = (
        "You are a verifier refiner. Produce improved child verifier bundles by editing the current bundle. "
        "Each child should be a small deterministic Python verifier bundle following the required contract."
    )
    user = (
        f"Task description:\n{task_description}\n\n"
        f"Current verifier bundle:\n```python\n{node.program.source_code}\n```\n\n"
        f"Critic summary:\n{critic_summary}\n\n"
        f"False positives:\n{format_examples_for_prompt(false_pos, include_labels=True)}\n\n"
        f"False negatives:\n{format_examples_for_prompt(false_neg, include_labels=True)}\n\n"
        f"Produce up to {num_children} improved child verifier bundles.\n\n"
        f"{VERIFIER_CONTRACT}\n\n"
        "Important refinement rules:\n"
        "- Keep bundles small.\n"
        "- Each verifier must declare a precise 'requires' list.\n"
        "- Prefer improving or replacing weak checks rather than only adding more checks.\n"
        "- Use context fields consistently.\n"
        "- If a stronger context field is needed, add it to the requires list.\n"
        "- Return only Python code blocks.\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_context_messages(
    task_description: str,
    x: object,
    y: object,
    required_fields: List[str],
    verifier_specs: List[dict],
) -> List[dict[str, str]]:
    system = CONTEXT_BUILDER_CONTRACT
    user = (
        f"Task description:\n{task_description}\n\n"
        f"Input x:\n{json.dumps(x, ensure_ascii=False, indent=2)}\n\n"
        f"Model output y:\n{json.dumps(y, ensure_ascii=False, indent=2)}\n\n"
        f"Required fields:\n{json.dumps(required_fields, ensure_ascii=False, indent=2)}\n\n"
        f"Verifier specs:\n{json.dumps(verifier_specs, ensure_ascii=False, indent=2)}\n\n"
        "Return ONLY a JSON object with exactly those required fields."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
