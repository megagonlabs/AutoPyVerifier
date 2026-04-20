# AutoPyVerifier: Learning Compact Executable Verifiers for Large Language Model Outputs
This repository is the implementation for the paper "AutoPyVerifier: Learning Compact Executable Verifiers for Large Language Model Outputs".

![alt text](https://github.com/megagonlabs/AutoPyVerifier/blob/main/figs/overview.png)

---

# AutoVerifier

AutoVerifier is a pipeline for searching over **deterministic Python verifier bundles** for labeled LLM outputs.

Given a development set of `(query, model_output, objective)` examples and a task description, the system uses an LLM to iteratively:

1. propose initial verifier bundles,
2. critique their failures,
3. refine them into new candidates,
4. execute each bundle in a restricted sandbox,
5. search over the DAG, and
6. select a compact verifier set that best balances the score, exploration, feasibility, and size.

---

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
    └── toy/
        ├── devset.jsonl
        └── task_description.txt

```

## Requirements

Use Python `>= 3.10.18`.

Install dependencies:

```bash
pip install -r requirements.txt
```

## API keys

Set only the key for the backend you plan to use. For example:

```bash
export OPENAI_API_KEY="..."
```

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

- what `query` and `output` represent,
- what objective labels `1` and `0` mean,
- what kinds of verifier logic are allowed or desired, and
- what the search should optimize for.

## Quick start

### 1. Run the toy example

From the project root:

```bash
python -m autopyverifier.cli search \
  --devset data/toy/devset.jsonl \
  --task_description_file data/toy/task_description.txt \
  --llm_backend openai \
  --seed_model gpt-5.4 \
  --critic_model gpt-5.4 \
  --refine_model gpt-5.4 \
  --context_model gpt-5.4 \
  --budget 20 \
  --feasible_coef 0.1 \
  --explore_coef 0.1 \
  --size_coef 0.1 \
  --out_dir results/toy/gpt54
```

Useful optional flags:

- `--budget`: number of search iterations
- `--temperature`: sampling temperature passed to the backend
- `--max_output_tokens`: output token budget for model calls
- `--beta_pp`: feasibility threshold for lower-confidence acceptance precision
- `--beta_np`: feasibility threshold for lower-confidence rejection precision
- `--timeout_seconds`: per-bundle execution timeout
- `--out_dir`: where to write search artifacts

## What gets written to `out_dir`

When `--out_dir` is provided, the search writes:

- `selected_verifier.py`: source code for the chosen verifier bundle
- `selected_verifier.json`: summary of the chosen verifier
- `graph.json`: metadata for all explored nodes
- `summary.json`: high-level search summary
- `nodes/*.py`: source code for each explored verifier bundle


## ⭐ **Citation**

If you would like to cite our work, the bibtex is:

    @article{pezeshkpour2026autopyverifier,
    title={AutoPyVerifier: Learning Compact Executable Verifiers for Large Language Model Outputs},
    author={Pezeshkpour, Pouya and Hruschka, Estevam},
    year={2026}
    }

---

## 📜 **Disclosure**
Embedded in, or bundled with, this product are open source software (OSS) components, datasets and other third party components identified below. The license terms respectively governing the datasets and third-party components continue to govern those portions, and you agree to those license terms, which, when applicable, specifically limit any distribution. You may receive a copy of, distribute and/or modify any open source code for the OSS component under the terms of their respective licenses, which may be CC license and Apache 2.0 license. In the event of conflicts between Megagon Labs, Inc., license conditions and the Open Source Software license conditions, the Open Source Software conditions shall prevail with respect to the Open Source Software portions of the software. You agree not to, and are not permitted to, distribute actual datasets used with the OSS components listed below. You agree and are limited to distribute only links to datasets from known sources by listing them in the datasets overview table below. You are permitted to distribute derived datasets of data sets from known sources by including links to original dataset source in the datasets overview table below. You agree that any right to modify datasets originating from parties other than Megagon Labs, Inc. are governed by the respective third party’s license conditions. All OSS components and datasets are distributed WITHOUT ANY WARRANTY, without even implied warranty such as for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE, and without any liability to or claim against any Megagon Labs, Inc. entity other than as explicitly documented in this README document. You agree to cease using any part of the provided materials if you do not agree with the terms or the lack of any warranty herein. While Megagon Labs, Inc., makes commercially reasonable efforts to ensure that citations in this document are complete and accurate, errors may occur. If you see any error or omission, please help us improve this document by sending information to contact_oss@megagon.ai.
