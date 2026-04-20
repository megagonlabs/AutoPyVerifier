from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class Example:
    id: str
    x: Any
    y: Any
    objective: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AtomicVerifier:
    name: str
    code: str
    description: str = ""
    requires: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Aggregator:
    name: str
    code: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierSetProgram:
    source_code: str
    verifiers: List[AtomicVerifier]
    aggregator: Aggregator
    signature: str

    @property
    def size(self) -> int:
        return len(self.verifiers)

    @property
    def required_fields(self) -> List[str]:
        fields: Set[str] = set()
        for verifier in self.verifiers:
            fields.update(verifier.requires)
        return sorted(fields)


@dataclass
class EvalStats:
    tp: int
    tn: int
    accept: int
    reject: int
    pp: float
    np: float
    cov_pos: float
    cov_neg: float
    feasible: bool
    score: float
    lcb_pp: float = 0.0
    lcb_np: float = 0.0
    false_pos_ids: List[str] = field(default_factory=list)
    false_neg_ids: List[str] = field(default_factory=list)
    accepted_ids: List[str] = field(default_factory=list)
    rejected_ids: List[str] = field(default_factory=list)
    context_failures: List[str] = field(default_factory=list)


@dataclass
class Node:
    program: VerifierSetProgram
    parents: Set[str] = field(default_factory=set)
    children: Set[str] = field(default_factory=set)
    stats: Optional[EvalStats] = None
    critic_summary: str = ""
    visits: int = 0


@dataclass
class SearchResult:
    selected_signature: str
    selected_node: Node
    all_nodes: Dict[str, Node]
    feasible_signatures: List[str]
    pareto_signatures: List[str]
