from __future__ import annotations

import hashlib
import random
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from tqdm import tqdm

from .config import LLMConfig, SearchConfig
from .execution import evaluate_bundle_source, extract_python_blocks, parse_program_source
from .metrics import compute_metrics
from .models import EvalStats, Example, Node, SearchResult
from .prompts import (
    build_context_messages,
    build_critic_messages,
    build_refine_messages,
    build_seed_messages,
)


class AutoVerifierSearch:
    def __init__(
        self,
        *,
        task_description: str,
        devset: List[Example],
        seed_client,
        critic_client,
        refine_client,
        context_client,
        llm_config: LLMConfig,
        search_config: SearchConfig,
    ) -> None:
        self.task_description = task_description
        self.devset = devset
        self.example_by_id = {ex.id: ex for ex in devset}
        self.seed_client = seed_client
        self.critic_client = critic_client
        self.refine_client = refine_client
        self.context_client = context_client
        self.llm_config = llm_config
        self.cfg = search_config
        self.nodes: Dict[str, Node] = {}
        self.root_sig = "__ROOT__"
        self.context_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def run(self) -> SearchResult:
        self._init_root()
        self._seed_nodes()
        for _  in tqdm(range(self.cfg.budget)):
            current = self._select_node()
            if current is None:
                break
            current.visits += 1
            false_pos = self._example_subset(current.stats.false_pos_ids, self.cfg.max_failure_examples_in_prompt)
            false_neg = self._example_subset(current.stats.false_neg_ids, self.cfg.max_failure_examples_in_prompt)
            critic_summary = self._run_critic(current, false_pos, false_neg)
            current.critic_summary = critic_summary
            self._expand_node(current, critic_summary, false_pos, false_neg)

        selected = self._select_final_node()
        result = SearchResult(
            selected_signature=selected.program.signature,
            selected_node=selected,
            all_nodes=self.nodes,
            feasible_signatures=[
                sig for sig, node in self.nodes.items() if sig != self.root_sig and node.stats and node.stats.feasible
            ],
            pareto_signatures=[node.program.signature for node in self._pareto_frontier()],
        )
        self._maybe_write_outputs(result)
        return result

    def _init_root(self) -> None:
        root_program = parse_program_source(
            "VERIFIER_SPECS=[{'name':'noop','description':'root placeholder','requires':[]}]\n"
            "def noop(x,y,context=None):\n    return True\n"
            "def aggregate(checks,x,y,context=None):\n    return True\n"
        )
        root_program.signature = self.root_sig
        self.nodes[self.root_sig] = Node(program=root_program)

    def _seed_nodes(self) -> None:
        seed_examples = self.devset[: self.cfg.max_examples_in_prompt]
        messages = build_seed_messages(self.task_description, seed_examples, self.cfg.num_seeds)
        response = self.seed_client.complete(
            messages,
            model=self.llm_config.seed_model,
            temperature=self.llm_config.temperature,
            max_output_tokens=self.llm_config.max_output_tokens,
        )

        blocks = extract_python_blocks(response)
        inserted = 0
        for block in blocks:
            before = len(self.nodes)
            self._insert_or_link_child(self.root_sig, block)
            after = len(self.nodes)
            if after > before:
                inserted += 1
        if inserted == 0:
            raise RuntimeError(
                "Seed generator did not produce any valid verifier bundles. "
                "Check the seed-model output format and the root-generation prompt."
            )

    def _insert_or_link_child(self, parent_sig: str, source_code: str) -> None:
        try:
            program = parse_program_source(source_code)
        except Exception:
            return

        sig = program.signature
        if sig in self.nodes:
            self.nodes[parent_sig].children.add(sig)
            self.nodes[sig].parents.add(parent_sig)
            return
        node = Node(program=program, visits=1)
        node.parents.add(parent_sig)
        self.nodes[parent_sig].children.add(sig)
        node.stats = self._evaluate_program(program)
        self.nodes[sig] = node

    def _required_signature(self, required_fields: List[str]) -> str:
        return "|".join(sorted(required_fields))

    def _build_context_for_example(self, ex: Example, program) -> Dict[str, Any]:
        required_fields = program.required_fields
        if not required_fields:
            return {}
        cache_key = (ex.id, self._required_signature(required_fields))
        if cache_key in self.context_cache:
            return self.context_cache[cache_key]

        verifier_specs = [
            {"name": v.name, "description": v.description, "requires": v.requires}
            for v in program.verifiers
        ]
        messages = build_context_messages(
            self.task_description,
            ex.x,
            ex.y,
            required_fields,
            verifier_specs,
        )
        context = self.context_client.build_context(
            messages,
            model=self.llm_config.context_model,
            temperature=0.0,
            max_output_tokens=2048,
        )
        if not isinstance(context, dict):
            raise ValueError("Context builder did not return a JSON object")
        # Guarantee all required fields exist.
        normalized = {field: context.get(field, None) for field in required_fields}
        self.context_cache[cache_key] = normalized
        return normalized

    def _evaluate_program(self, program) -> EvalStats:
        payload: List[dict] = []
        context_failures: List[str] = []
        for ex in self.devset:
            try:
                context = self._build_context_for_example(ex, program)
            except Exception as exc:
                context_failures.append(ex.id)
                context = {field: None for field in program.required_fields}

            payload.append(
                {
                    "id": ex.id,
                    "x": ex.x,
                    "y": ex.y,
                    "context": context,
                }
            )

        ok, bundle_results = evaluate_bundle_source(
            program.source_code,
            payload,
            timeout_seconds=self.cfg.timeout_seconds,
        )
        if not ok:
            return EvalStats(
                tp=0,
                tn=0,
                accept=0,
                reject=0,
                pp=0.0,
                np=0.0,
                cov_pos=0.0,
                cov_neg=0.0,
                feasible=False,
                score=-1e9,
                lcb_pp=0.0,
                lcb_np=0.0,
                context_failures=context_failures,
            )

        tp = tn = accept = reject = 0
        accepted_ids: List[str] = []
        rejected_ids: List[str] = []
        false_pos_ids: List[str] = []
        false_neg_ids: List[str] = []

        for row in bundle_results:
            ex = self.example_by_id[row["id"]]
            if row["accept"]:
                accept += 1
                accepted_ids.append(ex.id)
                if ex.objective == 1:
                    tp += 1
                else:
                    false_pos_ids.append(ex.id)
            else:
                reject += 1
                rejected_ids.append(ex.id)
                if ex.objective == 0:
                    tn += 1
                else:
                    false_neg_ids.append(ex.id)

        pp, npv, cov_pos, cov_neg, feasible, score, lcb_pp, lcb_np = compute_metrics(
            tp=tp,
            tn=tn,
            accept=accept,
            reject=reject,
            n=len(self.devset),
            beta_pp=self.cfg.beta_pp,
            beta_np=self.cfg.beta_np,
            set_size=program.size,
            delta=self.cfg.delta,
        )

        return EvalStats(
            tp=tp,
            tn=tn,
            accept=accept,
            reject=reject,
            pp=pp,
            np=npv,
            cov_pos=cov_pos,
            cov_neg=cov_neg,
            feasible=feasible,
            score=score,
            false_pos_ids=false_pos_ids,
            false_neg_ids=false_neg_ids,
            accepted_ids=accepted_ids,
            rejected_ids=rejected_ids,
            lcb_pp=lcb_pp,
            lcb_np=lcb_np,
            context_failures=context_failures,
        )

    def _example_subset(self, ids: List[str], limit: int) -> List[Example]:
        if limit <= 0 or not ids:
            return []        
        if len(ids) <= limit:
            return [self.example_by_id[i] for i in ids]
        rng = random.Random(1234)
        chosen_ids = rng.sample(ids, limit)
        return [self.example_by_id[i] for i in chosen_ids]

    def _run_critic(self, node: Node, false_pos: List[Example], false_neg: List[Example]) -> str:
        messages = build_critic_messages(self.task_description, node, false_pos, false_neg)
        return self.critic_client.complete(
            messages,
            model=self.llm_config.critic_model,
            temperature=self.llm_config.temperature,
            max_output_tokens=self.llm_config.max_output_tokens,
        )

    def _expand_node(self, node: Node, critic_summary: str, false_pos: List[Example], false_neg: List[Example]) -> None:
        messages = build_refine_messages(
            self.task_description,
            node,
            critic_summary,
            false_pos,
            false_neg,
            self.cfg.refine_children,
        )
        response = self.refine_client.complete(
            messages,
            model=self.llm_config.refine_model,
            temperature=self.llm_config.temperature,
            max_output_tokens=self.llm_config.max_output_tokens,
        )
        for block in extract_python_blocks(response):
            self._insert_or_link_child(node.program.signature, block)


    def _select_node(self) -> Node | None:
        candidates = [node for sig, node in self.nodes.items() if sig != self.root_sig]
        if not candidates:
            return None

        total_visits = sum(n.visits for n in candidates)

        def acquisition(n: Node) -> float:
            feasible_bonus = 1.0 if n.stats and n.stats.feasible else 0.0
            score_term = n.stats.score if n.stats else -1e9
            size_penalty = n.program.size
            explore_term = math.sqrt(math.log(1 + total_visits) / (1 + n.visits))
            return score_term + self.cfg.feasible_coef * feasible_bonus + self.cfg.explore_coef * explore_term - self.cfg.size_coef * size_penalty

        return max(candidates, key=acquisition)

    @staticmethod
    def _dominates(a: Node, b: Node) -> bool:
        a_tuple = (a.stats.lcb_pp, a.stats.lcb_np, a.stats.score, -a.program.size)
        b_tuple = (b.stats.lcb_pp, b.stats.lcb_np, b.stats.score, -b.program.size)
        return all(x >= y for x, y in zip(a_tuple, b_tuple)) and any(x > y for x, y in zip(a_tuple, b_tuple))

    def _pareto_frontier(self) -> List[Node]:
        vals = [node for sig, node in self.nodes.items() if sig != self.root_sig]
        return [n for n in vals if not any(m is not n and self._dominates(m, n) for m in vals)]

    def _select_final_node(self) -> Node:
        feasible_nodes = [
            node for sig, node in self.nodes.items() if sig != self.root_sig and node.stats and node.stats.feasible
        ]
        if feasible_nodes:
            feasible_nodes.sort(
                key=lambda n: (-n.stats.score, n.program.size, -n.stats.pp, -n.stats.np)
            )
            return feasible_nodes[0]
        frontier = self._pareto_frontier()
        if not frontier:
            raise RuntimeError("No verifier nodes were successfully evaluated")
        return max(frontier, key=lambda n: n.stats.score)

    def _maybe_write_outputs(self, result: SearchResult) -> None:
        if not self.cfg.out_dir:
            return
        out_dir = Path(self.cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def _safe_sig(sig: str) -> str:
            return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]

        selected_path = out_dir / "selected_verifier.py"
        selected_path.write_text(result.selected_node.program.source_code, encoding="utf-8")

        selected_meta = {
            "selected_signature": result.selected_signature,
            "size": result.selected_node.program.size,
            "required_fields": result.selected_node.program.required_fields,
            "verifier_names": [v.name for v in result.selected_node.program.verifiers],
            "aggregator": result.selected_node.program.aggregator.name,
            "pp": result.selected_node.stats.pp,
            "np": result.selected_node.stats.np,
            "lcb_pp": result.selected_node.stats.lcb_pp,
            "lcb_np": result.selected_node.stats.lcb_np,
            "cov_pos": result.selected_node.stats.cov_pos,
            "cov_neg": result.selected_node.stats.cov_neg,
            "feasible": result.selected_node.stats.feasible,
            "score": result.selected_node.stats.score,
        }
        (out_dir / "selected_verifier.json").write_text(json.dumps(selected_meta, indent=2), encoding="utf-8")

        nodes_dir = out_dir / "nodes"
        nodes_dir.mkdir(parents=True, exist_ok=True)
        for sig, node in self.nodes.items():
            if sig == self.root_sig:
                continue
            safe_name = _safe_sig(sig)
            (nodes_dir / f"{safe_name}.py").write_text(node.program.source_code, encoding="utf-8")

        graph = {}
        for sig, node in self.nodes.items():
            if sig == self.root_sig:
                continue
            safe_name = _safe_sig(sig)
            graph[sig] = {
                "parents": sorted(node.parents),
                "children": sorted(node.children),
                "size": node.program.size,
                "required_fields": node.program.required_fields,
                "verifier_names": [v.name for v in node.program.verifiers],
                "aggregator": node.program.aggregator.name,
                "feasible": bool(node.stats.feasible) if node.stats else False,
                "score": float(node.stats.score) if node.stats else None,
                "pp": float(node.stats.pp) if node.stats else None,
                "np": float(node.stats.np) if node.stats else None,
                "lcb_pp": float(node.stats.lcb_pp) if node.stats else None,
                "lcb_np": float(node.stats.lcb_np) if node.stats else None,
                "cov_pos": float(node.stats.cov_pos) if node.stats else None,
                "cov_neg": float(node.stats.cov_neg) if node.stats else None,
                "context_failures": list(node.stats.context_failures) if node.stats else [],
                "code_file": f"nodes/{safe_name}.py",
            }
        (out_dir / "graph.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")

        summary = {
            "selected_signature": result.selected_signature,
            "feasible_signatures": result.feasible_signatures,
            "pareto_signatures": result.pareto_signatures,
            "num_nodes": len([sig for sig in self.nodes if sig != self.root_sig]),
            "num_context_cache_entries": len(self.context_cache),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
