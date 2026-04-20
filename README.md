# AutoVerifier

AutoVerifier is a small research-style pipeline for searching over **deterministic Python verifier bundles** for labeled LLM outputs.

Given a development set of `(query, model_output, objective)` examples and a task description, the system uses an LLM to:

1. propose initial verifier bundles,
2. critique their failures,
3. refine them into new candidates,
4. execute each bundle in a restricted sandbox, and
5. select a compact verifier set that best balances acceptance precision, rejection precision, coverage, and size.

The codebase is organized as a lightweight CLI package and currently supports `mock`, `openai`, `gemini`, and `claude` backends.

## What the project does

At a high level, AutoVerifier searches for a verifier set of the following form:

- a `VERIFIER_SPECS` list describing atomic verifier functions,
- one Python function per verifier, and
- an `aggregate(checks, x, y, context=None)` function that combines verifier decisions into a final accept/reject verdict.

Each candidate bundle is evaluated on a labeled development set. The search keeps track of candidate nodes, expands promising ones, and writes the selected verifier plus search artifacts to disk.

## Repository structure

```text
.
├── autoverifier/
│   ├── cli.py              # CLI entry point
│   ├── config.py           # search and model configuration dataclasses
│   ├── data.py             # JSONL devset loading utilities
│   ├── execution.py        # verifier parsing, sandboxing, execution
│   ├── metrics.py          # scoring and feasibility metrics
│   ├── models.py           # shared dataclasses
│   ├── prompts.py          # seed / critic / refine / context prompts
│   ├── search.py           # main single-DAG search loop
│   └── llm/
│       ├── base.py
│       ├── mock.py
│       ├── openai_llms.py
│       ├── gemini_llms.py
│       └── claude_llms.py
├── data/
│   └── toy/
│       ├── devset.jsonl
│       └── task_description.txt
└── run.sh                  # example run script
```

## Requirements

Use Python `>= 3.10.18`.

Install dependencies:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` for the current codebase is:

```txt
# Requires Python >= 3.10.18
openai
anthropic
google-genai
tqdm
```

## API keys

Set only the key for the backend you plan to use:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GEMINI_API_KEY="..."
```

Notes:

- The OpenAI backend expects the `openai` Python SDK.
- The Gemini backend expects `google-genai`.
- The Claude backend expects `anthropic`.
- In the current implementation, backend modules are imported eagerly, so having all listed SDK packages installed is the safest setup even if you only plan to use one backend.

## Input format

The development set is a JSONL file. Each line should have:

- `id`: example identifier
- `query`: task input
- `output`: model output to verify
- `objective`: `1` if the output satisfies the target objective, else `0`
- `metadata` (optional): extra per-example metadata

Example:

```json
{"id": "m1", "query": "Solve x^2 - 5x + 6 = 0.", "output": "x^2 - 5x + 6 = (x-2)(x-3), so x=2 or x=3.", "objective": 1}
{"id": "m2", "query": "Solve x^2 - 5x + 6 = 0.", "output": "The roots are 1 and 6.", "objective": 0}
```

The task description is a plain-text file describing:

- what `x` and `y` represent,
- what objective label `1` means,
- what kinds of verifier logic are allowed or desired, and
- what the search should optimize for.

## Quick start

### 1. Run the toy example with a real backend

From the project root:

```bash
python -m autoverifier.cli search \
  --devset data/toy/devset.jsonl \
  --task_description_file data/toy/task_description.txt \
  --llm_backend openai \
  --seed_model gpt-5.4 \
  --critic_model gpt-5.4 \
  --refine_model gpt-5.4 \
  --context_model gpt-5.4 \
  --budget 2 \
  --feasible_coef 0.1 \
  --explore_coef 0.1 \
  --size_coef 0.1 \
  --out_dir results/toy/gpt54
```

This is essentially what `run.sh` demonstrates, except you should supply the API key through environment variables rather than hardcoding it in the script.

### 2. Run with the mock backend

If you only want to exercise the search flow without external API calls:

```bash
python -m autoverifier.cli search \
  --devset data/toy/devset.jsonl \
  --task_description_file data/toy/task_description.txt \
  --llm_backend mock \
  --seed_model mock-seed \
  --critic_model mock-critic \
  --refine_model mock-refine \
  --context_model mock-context \
  --budget 2 \
  --out_dir results/toy/mock
```

## Main CLI arguments

```text
python -m autoverifier.cli search \
  --devset PATH \
  --task_description_file PATH | --task_description TEXT \
  --llm_backend {mock,openai,gemini,claude} \
  --seed_model MODEL \
  --critic_model MODEL \
  --refine_model MODEL \
  --context_model MODEL
```

Useful optional flags:

- `--budget`: number of search iterations
- `--temperature`: sampling temperature passed to the backend
- `--max_output_tokens`: output token budget for model calls
- `--beta_pp`: feasibility threshold for lower-confidence acceptance precision
- `--beta_np`: feasibility threshold for lower-confidence rejection precision
- `--feasible_coef`: acquisition bonus for feasible nodes
- `--explore_coef`: exploration weight in node selection
- `--size_coef`: penalty on verifier-set size
- `--timeout_seconds`: per-bundle execution timeout
- `--out_dir`: where to write search artifacts

## What gets written to `out_dir`

When `--out_dir` is provided, the search writes:

- `selected_verifier.py`: source code for the chosen verifier bundle
- `selected_verifier.json`: summary of the chosen verifier
- `graph.json`: metadata for all explored nodes
- `summary.json`: high-level search summary
- `nodes/*.py`: source code for each explored verifier bundle

## How evaluation works

For each candidate verifier bundle:

1. the system determines which `context` fields are required from `VERIFIER_SPECS`,
2. an LLM context builder produces those fields for each example,
3. the verifier bundle is executed in a subprocess sandbox,
4. each example is accepted or rejected by `aggregate(...)`, and
5. the bundle is scored using precision-oriented metrics and feasibility thresholds.

The current metrics implementation computes:

- positive-side precision (`PP`),
- negative-side precision (`NP`),
- accept and reject coverage,
- lower confidence bounds for `PP` and `NP`, and
- a macro-F1-style score used inside search.

A node is considered feasible when its lower confidence bounds clear the configured `beta_pp` and `beta_np` thresholds.

## Sandbox restrictions for generated verifiers

Verifier code is intentionally restricted.

Allowed imports are limited to:

- `math`
- `re`
- `json`
- `statistics`
- `fractions`
- `decimal`
- `itertools`
- `collections`
- `ast`

The sandbox also blocks several risky builtins and syntax forms such as `eval`, `exec`, `open`, `input`, class definitions, async constructs, and context managers.

## Notes and current limitations

- This is a compact experimental codebase rather than a packaged library.
- The current search implementation is a single-DAG search rooted at a placeholder node.
- `run.sh` should not contain hardcoded secrets; prefer environment variables.
- Because backend modules are imported eagerly, missing SDK packages can cause import-time failures even when that backend is not selected.
- Generated verifier bundles must return Python code blocks in the expected format, or the candidate will be ignored.

## Suggested first improvements

If you plan to keep developing this repository, these are sensible next steps:

- add a `pyproject.toml` and package metadata,
- move API-key handling entirely to environment variables,
- make backend imports lazy so users only install the SDK they need,
- add tests for parsing, sandbox validation, and metric computation, and
- add a richer example beyond the toy math dataset.

## License

Add your preferred license here before publishing.
# AutoPyVerifier
