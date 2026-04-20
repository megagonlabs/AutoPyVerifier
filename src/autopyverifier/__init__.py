"""Public package interface for AutoPyVerifier."""

from . import config, data, execution, metrics, models, prompts, search
from .config import LLMConfig, SearchConfig
from .data import format_examples_for_prompt, load_devset
from .models import (
    Aggregator,
    AtomicVerifier,
    EvalStats,
    Example,
    SearchResult,
    VerifierSetProgram,
)
from .search import AutoVerifierSearch

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "config",
    "data",
    "execution",
    "metrics",
    "models",
    "prompts",
    "search",
    "Aggregator",
    "AtomicVerifier",
    "AutoVerifierSearch",
    "EvalStats",
    "Example",
    "LLMConfig",
    "SearchConfig",
    "SearchResult",
    "VerifierSetProgram",
    "format_examples_for_prompt",
    "load_devset",
]
