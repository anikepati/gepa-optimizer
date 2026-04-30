# Doc Extract v3 — Autonomous Prompt Optimizer on Google ADK

A production-grade, autonomous prompt optimization system for document extraction. Built on Google ADK with config-driven strategies, three-signal scoring, Pareto-archived candidate selection, and full OpenShift deployment.

This repo is the v3 evolution of the v2 doc-extract pipeline. The v2 extraction pipeline is unchanged and still runs from `main.py` against any extraction schema. v3 adds an *optimizer* on top: given a base schema and a labeled dataset, it autonomously evolves the schema until the operator's spec is met.

---

## What v3 does

You give it:
- A base extraction schema YAML (the starting point — fields, rules, examples)
- A labeled dataset (documents + expected JSON outputs)
- An OptimizationSpec (target metrics, budgets, iteration caps)

It produces:
- An evolved schema YAML that meets the spec, plus a Pareto frontier of trade-off variants
- A complete audit trail in Postgres (lineage, costs, every iteration)
- Live observability via the ADK web UI

Fully autonomous — no human approvals between iterations. Soft budget cap (warns and continues) so you never lose progress to a budget overrun.

---

## Architecture

```
prompt_optimizer (SequentialAgent)
├── bootstrap_eval         (one-shot: evaluate base schema, seed archive)
└── ralph_loop             (LoopAgent, max=R iterations)
    ├── strategy_selector  (rule-based, picks next mutation strategy)
    ├── ralph_iter_tracker (DB row)
    ├── gepa_loop          (LoopAgent, max=G iterations)
    │   ├── gepa_iter_tracker
    │   ├── reflection_agent      (LLM hypothesis from failures)
    │   ├── proposal_agent        (delegates to current strategy)
    │   ├── eval_fanout           (ParallelAgent, K branches)
    │   │   └── candidate_evaluator_<i>  (extracts, F1, judge, validators)
    │   ├── pareto_archive_agent  (merges, snapshots, picks best)
    │   └── gepa_convergence      (escalates inner loop)
    └── spec_validator     (escalates outer loop on satisfaction or wall clock)
```

### State namespacing

All session state uses prefixed keys:
- `run:*` — immutable run config
- `ralph:*` — outer loop state
- `gepa:*` — inner loop state
- `gepa:candidates:<slot>:results` — slot-specific evaluator output (avoids ParallelAgent races)
- `cost:*` — tracking
- `audit:*` — events

### Three-signal scoring

Each candidate is scored on three independent objectives:
1. **F1** (`mean_field_f1`) — deterministic per-field comparison
2. **LLM judge** (`mean_judge_score`) — semantic equivalence on F1 misses
3. **Custom validators** (`validator_pass_rate`) — pluggable Python rules (arithmetic, regex, date order)

The Pareto archive keeps the non-dominated frontier across all three.

### Strategies (config-driven)

Four built-in strategies, each a YAML descriptor + agent class:

| Strategy | When | What it does |
|----------|------|--------------|
| `reflection_mutation` | Default | LLM proposes targeted edits via 4 lenses |
| `demo_bootstrap` | Few-shot count low | Generates new few-shot demonstrations |
| `schema_decomposition` | Field plateaus < 0.7 | Splits a field into narrower sub-fields |
| `template_restructure` | Stagnation 2+ rounds | Mutates the prompt template, not the schema |

Add a strategy: drop a YAML in `config/strategies/` and a class in `optimizer/strategies/`. The strategy_selector finds it via the registry.

---

## Layout

```
doc-extract-v3/
├── core/                       # v2 pipeline (unchanged): schema, prompt builder, extractor, storage
├── agents/                     # v2 extraction agents (unchanged)
├── main.py                     # v2 extraction console app (unchanged)
├── optimizer/
│   ├── spec.py                 # OptimizationSpec dataclass
│   ├── pareto.py               # Domination + archive logic
│   ├── state_keys.py           # Namespaced state keys
│   ├── llm_client.py           # LLM factory + cost tracking
│   ├── orchestrator.py         # Builds the agent graph
│   ├── cli.py                  # start | status | promote
│   ├── web.py                  # ADK web UI entrypoint
│   ├── persistence/            # SQLAlchemy models, repo, db helpers
│   ├── agents/                 # All ADK agents in the topology
│   ├── strategies/             # 4 strategies + base/registry
│   ├── scoring/                # F1, judge, validators
│   └── callbacks/              # Budget warner, cost recorder
├── config/
│   ├── extraction_schema.yaml  # v2: base schema (input to optimizer)
│   ├── pipeline.yaml           # v2: OCR + storage settings
│   ├── optimization.yaml       # v3: spec, LLM models, budgets
│   └── strategies/             # 4 strategy descriptors
├── prompts/
│   └── extraction_prompt.yaml  # v2 template (still used)
├── dataset/
│   ├── documents/              # source files
│   └── expected/               # expected JSON ground truth
└── deploy/
    ├── Dockerfile
    └── ocp/                    # OpenShift manifests
```

---

## Quickstart

### 1. Local

```bash
# Install
pip install -r requirements.txt

# Provision Postgres (Docker or your existing instance)
docker run -d --name optimizer-db -p 5432:5432 \
  -e POSTGRES_DB=optimizer -e POSTGRES_USER=optimizer -e POSTGRES_PASSWORD=secret \
  postgres:16-alpine

export OPTIMIZER_DATABASE_URL="postgresql+psycopg2://optimizer:secret@localhost:5432/optimizer"
export GOOGLE_API_KEY="..."
# Optional:
export ANTHROPIC_API_KEY="..."
```

### 2. Edit the spec

`config/optimization.yaml` — set targets, budget, iteration counts. Defaults are realistic for ~30-doc datasets:

```yaml
spec:
  targets:
    - name: mean_field_f1
      threshold: 0.92
  budget:
    max_dollars: 50
    max_wall_clock_seconds: 14400
  ralph_max_iterations: 5
  gepa_max_iterations: 5
  candidates_per_gepa_iteration: 4
```

### 3. Run

```bash
python -m optimizer.cli start \
  --optimization-config config/optimization.yaml \
  --extraction-config config/extraction_schema.yaml \
  --pipeline-config config/pipeline.yaml \
  --dataset-dir dataset
```

Output: `Run complete: run-abc123def456 / Status: satisfied / Best candidate: cand-xyz`

### 4. Inspect

```bash
python -m optimizer.cli status run-abc123
# Shows costs, archive size, per-candidate metrics

# Or use the ADK web UI for live observation
adk web .
# → http://localhost:8000
```

### 5. Promote

```bash
python -m optimizer.cli promote run-abc123 cand-xyz \
  --output config/extraction_schema_optimized.yaml

# Now use the optimized schema with the v2 pipeline
python main.py --schema config/extraction_schema_optimized.yaml --inputs dataset/documents/
```

---

## Run flow detail

A run proceeds:

1. **Bootstrap.** Evaluates the base schema once on the dev set. Result becomes Pareto archive entry #0 and seeds reflection's failure context.
2. **Ralph iteration N starts.** `strategy_selector` picks one of the 4 strategies based on history (rotation with stagnation override).
3. **GEPA inner loop iteration M starts.** `reflection_agent` writes a one-paragraph hypothesis. `proposal_agent` delegates to the chosen strategy → produces K candidate schemas.
4. **Parallel evaluation.** `eval_fanout` (ParallelAgent) launches K `candidate_evaluator` branches. Each evaluates one candidate against all dev docs in parallel (thread pool inside).
5. **Archive merge.** `pareto_archive_agent` collects slot results, updates the frontier, snapshots to DB, picks current best.
6. **Convergence check.** GEPA loop ends if max iterations or N rounds without improvement.
7. **Spec check.** Ralph loop ends if all targets met or max iterations or wall clock exceeded.
8. **Best candidate** in the final archive is promoted via the `promote` CLI command.

### Resumability

ADK's `DatabaseSessionService` persists session state to Postgres. Combined with our optimizer tables, a pod restart resumes the run mid-iteration. `RUN_ID` is the session ID — the same value across restarts.

### Budget enforcement (soft)

Per spec: warn-and-continue. The cost ledger records every LLM call. The budget callback logs a warning when over the dollar cap but does NOT block. Wall-clock IS hard-enforced (terminates the Ralph loop).

---

## OpenShift deployment

```bash
# 1. Build and push image
docker build -t your-registry/doc-extract-optimizer:latest -f deploy/Dockerfile .
docker push your-registry/doc-extract-optimizer:latest

# 2. Create namespace + secrets
oc new-project doc-extract
oc create secret generic doc-extract-secrets \
  --from-literal=GOOGLE_API_KEY=$GOOGLE_API_KEY \
  --from-literal=ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  --from-literal=OPTIMIZER_DATABASE_URL="postgresql+psycopg2://optimizer:secret@doc-extract-postgres:5432/optimizer"

# 3. Create configmap from your config files
oc create configmap doc-extract-config \
  --from-file=config/optimization.yaml \
  --from-file=config/extraction_schema.yaml \
  --from-file=config/pipeline.yaml \
  --from-file=prompts/extraction_prompt.yaml

# 4. Apply manifests
oc apply -f deploy/ocp/

# 5. Verify
oc get pods
oc get route doc-extract-optimizer-web
```

The web UI is exposed via the OpenShift Route. Visit the URL for the live ADK developer interface — agent traces, state inspection, event timeline.

To kick off a run:

```bash
oc exec -it deploy/doc-extract-optimizer-worker -- \
  python -m optimizer.cli start \
    --optimization-config /app/config/optimization.yaml \
    --extraction-config /app/config/extraction_schema.yaml \
    --pipeline-config /app/config/pipeline.yaml \
    --dataset-dir /workspace/dataset
```

For scheduled autonomous runs, swap the worker `Deployment` for a `CronJob`.

### Production notes

- The included `postgres-statefulset.yaml` is for dev/standalone clusters. Use a managed Postgres (RDS, Cloud SQL, CrunchyData operator) in prod.
- Worker `replicas: 1` and `strategy: Recreate` are intentional — single writer per run.
- The `network-policy.yaml` allows TCP/443 egress (for LLM APIs); tighten with a dedicated egress proxy if your security posture requires it.

---

## Cost realism

A single full run with default config:

```
ralph_iter × gepa_iter × candidates × dev_docs = 5 × 5 × 4 × 12 = 1200 extraction calls
+ judge calls on incorrect fields (≈30% of fields × 1200) = ~360 judge calls
+ reflection + proposal calls = ~50 additional LLM calls
```

At Gemini Flash pricing (~$0.075/1M input, $0.30/1M output): ~$5–15 per run.

At Gemini 1.5 Pro for the proposer + Flash for everything else: ~$15–40 per run.

The default config uses **Flash for extraction and judge** (heavy) and **Pro for the proposer** (light, where reasoning quality matters). Edit `config/optimization.yaml` to change.

---

## Adding a new strategy

1. Create `optimizer/strategies/my_strategy.py`:

```python
from optimizer.strategies.base import (
    CandidateProposal, Strategy, StrategyContext, register,
)

@register
class MyStrategy(Strategy):
    name = "my_strategy"

    def propose(self, ctx: StrategyContext) -> list[CandidateProposal]:
        # Use ctx.current_schema_yaml, ctx.field_scores, ctx.failures_by_field
        # Call ctx.llm_fn(system, user) for LLM-backed mutations
        # Return up to ctx.n_candidates proposals
        return []
```

2. Create `config/strategies/my_strategy.yaml`:

```yaml
name: my_strategy
description: "What it does"
trigger:
  conditions: []
params: {}
```

3. Register the import in `optimizer/strategies/base.py::list_strategies()`.

The selector will rotate through it. To control when it's picked, edit `optimizer/agents/strategy_selector.py::pick_strategy()` — or replace the rule-based selector with an LLM-driven one (the hook is documented in the file).

---

## Troubleshooting

**`OPTIMIZER_DATABASE_URL not set`** — Set the env var (see Quickstart). Required for both CLI and web modes.

**`ParallelAgent state races`** — Each candidate evaluator writes to a slot-specific key (`gepa:candidates:N:results`). The archive agent reads all slots after the parallel branch completes. If you add custom state writes inside parallel agents, follow this pattern.

**Schema validation errors after promote** — The optimizer ensures schemas remain syntactically valid (round-trip through PyYAML), but a strategy may produce a schema that breaks downstream consumers. Run a quick smoke test with `main.py` before promoting.

**Cost overrun warnings, no termination** — By design (soft cap). To enforce a hard cap, change `make_budget_warner` in `optimizer/callbacks/budget_callback.py` to return an LlmResponse instead of None. The wall-clock cap is already hard-enforced.

**ADK web UI shows wrong agent** — The web UI imports `optimizer/web.py` at startup. If you change configs, restart the web pod (`oc rollout restart deployment/doc-extract-optimizer-web`).

---

## What's not v3 (deferred)

- Active learning (auto-labeling new docs to expand dataset)
- Multi-objective spec (e.g., "satisfy F1 within Y latency budget")
- Per-field optimization (treating each field as an independent search)
- Gradient-based prompt optimization (TextGrad-style)

These are reasonable v4 directions. v3 already gives the core production loop and a clean place to add them.
