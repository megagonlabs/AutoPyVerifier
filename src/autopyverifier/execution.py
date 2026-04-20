from __future__ import annotations

import ast
import multiprocessing as mp
import re
from types import MappingProxyType
from typing import Any, Dict, List, Tuple

from .models import AtomicVerifier, Aggregator, Example, VerifierSetProgram

ALLOWED_IMPORTS = {
    "math",
    "re",
    "json",
    "statistics",
    "fractions",
    "decimal",
    "itertools",
    "collections",
    "ast"
}


class SandboxError(RuntimeError):
    pass



def extract_python_blocks(text: str) -> List[str]:
    pattern = re.compile(
        r"(?ms)^```(?:python)?[^\n]*\n(.*?)^```[ \t]*(?:\n|$)",
        re.IGNORECASE | re.MULTILINE,
    )
    return [m.strip() for m in pattern.findall(text)]

def _limited_import(name: str, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if root not in ALLOWED_IMPORTS:
        raise ImportError(f"Import of module '{name}' is not allowed")
    return __import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "print": print,
        "__import__": _limited_import,
        "callable": callable,
        "Exception": Exception,
        "SyntaxError": SyntaxError,
    }

BANNED_CALL_NAMES = {
    "eval",
    "exec",
    "open",
    "compile",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
}

BANNED_NODE_TYPES = (
    ast.Global,
    ast.Nonlocal,
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.With,
    ast.AsyncWith,
    ast.Raise,
)


def validate_source(source: str) -> ast.AST:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SandboxError(f"Syntax error in verifier source: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, BANNED_NODE_TYPES):
            raise SandboxError(f"Banned syntax in verifier source: {type(node).__name__}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in ALLOWED_IMPORTS:
                    raise SandboxError(f"Import '{alias.name}' is not allowed")
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", 1)[0]
            if root not in ALLOWED_IMPORTS:
                raise SandboxError(f"Import from '{module}' is not allowed")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BANNED_CALL_NAMES:
                raise SandboxError(f"Call to '{node.func.id}' is not allowed")
    return tree


def canonicalize_source(source: str) -> str:
    tree = validate_source(source)
    try:
        return ast.unparse(tree)
    except Exception:
        return source.strip()


def _exec_source(source: str) -> Dict[str, Any]:
    validate_source(source)
    env: Dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    exec(compile(source, "<verifier_bundle>", "exec"), env, env)
    return env


def parse_program_source(source: str) -> VerifierSetProgram:
    env = _exec_source(source)
    specs = env.get("VERIFIER_SPECS")
    if not isinstance(specs, list) or not specs:
        raise SandboxError("Verifier bundle must define a non-empty VERIFIER_SPECS list")
    verifiers: List[AtomicVerifier] = []
    for spec in specs:
        if not isinstance(spec, dict) or "name" not in spec:
            raise SandboxError("Each item in VERIFIER_SPECS must be a dict with a 'name'")
        name = str(spec["name"])
        if name not in env or not callable(env[name]):
            raise SandboxError(f"Verifier function '{name}' is missing or not callable")
        requires = spec.get("requires", [])
        if requires is None:
            requires = []
        if not isinstance(requires, list) or not all(isinstance(x, str) for x in requires):
            raise SandboxError(f"Verifier '{name}' has invalid requires list")
        verifiers.append(
            AtomicVerifier(
                name=name,
                code="",
                description=str(spec.get("description", "")),
                requires=list(requires),
                metadata={k: v for k, v in spec.items() if k not in {"name", "description", "requires"}},
            )
        )
    if "aggregate" not in env or not callable(env["aggregate"]):
        raise SandboxError("Verifier bundle must define callable aggregate(checks, x, y, context=None)")
    sig = canonicalize_source(source)
    return VerifierSetProgram(
        source_code=source,
        verifiers=verifiers,
        aggregator=Aggregator(name="aggregate", code=""),
        signature=sig,
    )


def _worker_evaluate(source: str, examples_payload: List[dict], queue: mp.Queue) -> None:
    try:
        env = _exec_source(source)
        specs = env["VERIFIER_SPECS"]
        verifier_names = [str(spec["name"]) for spec in specs]
        aggregate = env["aggregate"]
        results = []
        for ex in examples_payload:
            x = ex["x"]
            y = ex["y"]
            context = ex.get("context")
            checks = {}
            for name in verifier_names:
                verdict = env[name](x, y, context)
                checks[name] = bool(verdict)
            accept = bool(aggregate(checks, x, y, context))
            results.append({"id": ex["id"], "checks": checks, "accept": accept})
        queue.put({"ok": True, "results": results})
    except Exception as exc:  
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def evaluate_bundle_source(
    source: str,
    examples_payload: List[dict],
    timeout_seconds: float = 8.0,
) -> Tuple[bool, Any]:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_worker_evaluate, args=(source, examples_payload, queue))
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False, f"Timeout after {timeout_seconds:.1f}s"
    if queue.empty():
        return False, "Verifier subprocess produced no output"
    message = queue.get()
    if not message.get("ok", False):
        return False, message.get("error", "Unknown verifier execution failure")
    return True, message["results"]
