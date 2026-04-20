from __future__ import annotations

import argparse
from pathlib import Path

from .config import LLMConfig, SearchConfig
from .data import load_devset
from .llm import MockLLMClient, OpenAICompatibleClient, GeminiCompatibleClient, ClaudeCompatibleClient
from .search import AutoVerifierSearch


def build_client(backend: str):
    if backend == "mock":
        return MockLLMClient()
    if backend == "openai":
        return OpenAICompatibleClient()
    if backend == "gemini":
        return GeminiCompatibleClient()
    if backend == "claude":
        return ClaudeCompatibleClient()

    raise ValueError(f"Unknown LLM backend: {backend}")


def read_task_description(args) -> str:
    if args.task_description_file:
        return Path(args.task_description_file).read_text(encoding="utf-8")
    if args.task_description:
        return args.task_description
    raise ValueError("Provide --task-description or --task-description-file")


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoVerifier single-DAG search")
    sub = parser.add_subparsers(dest="command", required=True)

    search_p = sub.add_parser("search", help="Run verifier-set search")
    search_p.add_argument("--devset", required=True)
    search_p.add_argument("--task_description")
    search_p.add_argument("--task_description_file")
    search_p.add_argument("--llm_backend", default="mock", choices=["mock", "openai", "gemini", "claude"])
    search_p.add_argument("--seed_model", default="mock-seed")
    search_p.add_argument("--critic_model", default="mock-critic")
    search_p.add_argument("--refine_model", default="mock-refine")
    search_p.add_argument("--context_model", default="mock-context")
    search_p.add_argument("--temperature", type=float, default=None)
    search_p.add_argument("--max_output_tokens", type=int, default=None)
    search_p.add_argument("--budget", type=int, default=60)
    search_p.add_argument("--beta_pp", type=float, default=0.90)
    search_p.add_argument("--beta_np", type=float, default=0.90)
    search_p.add_argument("--feasible_coef", type=float, default=0.5)
    search_p.add_argument("--explore_coef", type=float, default=0.5)
    search_p.add_argument("--size_coef", type=float, default=0.1)
    search_p.add_argument("--timeout_seconds", type=float, default=8.0)
    search_p.add_argument("--out_dir")

    args = parser.parse_args()
    devset = load_devset(args.devset)
    task_description = read_task_description(args)

    client = build_client(args.llm_backend)
    llm_cfg = LLMConfig(
        backend=args.llm_backend,
        seed_model=args.seed_model,
        critic_model=args.critic_model,
        refine_model=args.refine_model,
        context_model=args.context_model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    search_cfg = SearchConfig(
        budget=args.budget,
        beta_pp=args.beta_pp,
        beta_np=args.beta_np,
        feasible_coef=args.feasible_coef,
        explore_coef=args.explore_coef,
        size_coef=args.size_coef,
        timeout_seconds=args.timeout_seconds,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )

    engine = AutoVerifierSearch(
        task_description=task_description,
        devset=devset,
        seed_client=client,
        critic_client=client,
        refine_client=client,
        context_client=client,
        llm_config=llm_cfg,
        search_config=search_cfg,
    )
    result = engine.run()

    print("Selected signature:", result.selected_signature)
    print("Selected size:", result.selected_node.program.size)
    print("Required fields:", result.selected_node.program.required_fields)
    print("Feasible:", result.selected_node.stats.feasible)
    print("PP:", round(result.selected_node.stats.pp, 4))
    print("NP:", round(result.selected_node.stats.np, 4))
    print("LCB_PP:", round(result.selected_node.stats.lcb_pp, 4))
    print("LCB_NP:", round(result.selected_node.stats.lcb_np, 4))
    print("Accept coverage:", round(result.selected_node.stats.cov_pos, 4))
    print("Reject coverage:", round(result.selected_node.stats.cov_neg, 4))
    print("Score:", round(result.selected_node.stats.score, 4))
    if search_cfg.out_dir:
        print("Artifacts written to:", search_cfg.out_dir)


if __name__ == "__main__":
    main()
