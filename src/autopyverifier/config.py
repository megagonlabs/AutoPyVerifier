from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SearchConfig:
    num_seeds: int = 3
    budget: int = 60
    beta_pp: float = 0.50
    beta_np: float = 0.50
    delta: float = 0.05
    refine_children: int = 3
    feasible_coef: float = 0.5
    explore_coef: float = 0.5
    size_coef: float = 0.1
    timeout_seconds: float = 60.0
    max_examples_in_prompt: int = 3
    max_failure_examples_in_prompt: int = 3
    max_context_examples_per_call: int = 1
    out_dir: Optional[Path] = None


@dataclass
class LLMConfig:
    temperature: None
    max_output_tokens: None
    backend: str = "mock"
    seed_model: str = "mock-seed"
    critic_model: str = "mock-critic"
    refine_model: str = "mock-refine"
    context_model: str = "mock-context"
