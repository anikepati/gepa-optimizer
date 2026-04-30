"""Optimizer CLI.

Commands:
  start  - Kick off a new optimization run.
  status - Inspect a run's current state, archive, costs.
  promote - Promote a candidate's schema to disk (replaces base schema or writes new file).

Examples:
  python -m optimizer.cli start \\
    --optimization-config config/optimization.yaml \\
    --extraction-config config/extraction_schema.yaml \\
    --pipeline-config config/pipeline.yaml \\
    --dataset-dir dataset

  python -m optimizer.cli status run-abc123

  python -m optimizer.cli promote run-abc123 cand-xyz \\
    --output config/extraction_schema_optimized.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from optimizer.orchestrator import build_optimizer_root_agent
from optimizer.persistence import init_engine, repo
from optimizer.spec import OptimizationSpec
from optimizer import state_keys as K

logger = logging.getLogger(__name__)


# ============================================================================
# Dataset loading
# ============================================================================

def load_dataset(dataset_dir: str | Path) -> list[dict]:
    """Load every (document, expected.json) pair from a dataset directory.

    Layout:
      dataset/
        documents/<name>.{pdf|docx|txt|...}
        expected/<name>.json

    Returns: [{name, path, expected}].
    """
    dataset_dir = Path(dataset_dir)
    docs_dir = dataset_dir / "documents"
    expected_dir = dataset_dir / "expected"
    if not docs_dir.is_dir() or not expected_dir.is_dir():
        raise FileNotFoundError(f"dataset_dir must contain documents/ and expected/: {dataset_dir}")

    out: list[dict] = []
    for doc_path in sorted(docs_dir.iterdir()):
        if doc_path.is_dir() or doc_path.name.startswith("."):
            continue
        name = doc_path.stem
        exp_path = expected_dir / f"{name}.json"
        if not exp_path.exists():
            logger.warning(f"no expected.json for {name}, skipping")
            continue
        try:
            expected = json.loads(exp_path.read_text())
        except json.JSONDecodeError as e:
            logger.warning(f"bad json for {name}: {e}")
            continue
        out.append({
            "name": name,
            "path": str(doc_path),
            "expected": expected,
        })
    return out


def split_dataset(
    examples: list[dict], dev_frac: float = 0.4, train_frac: float = 0.4,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Deterministic split. dev gets first ~40%, train next ~40%, test the rest.

    The split is based on alphabetical name order so it is reproducible
    across runs.
    """
    n = len(examples)
    if n < 3:
        # tiny dataset: use everything as dev
        return list(examples), [], []
    n_dev = max(1, int(n * dev_frac))
    n_train = max(1, int(n * train_frac))
    return examples[:n_dev], examples[n_dev:n_dev + n_train], examples[n_dev + n_train:]


# ============================================================================
# start command
# ============================================================================

def cmd_start(args) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load optimization config (spec + LLM + processing)
    opt_config_path = Path(args.optimization_config)
    if not opt_config_path.exists():
        print(f"ERROR: optimization config not found: {opt_config_path}", file=sys.stderr)
        return 2
    with open(opt_config_path) as f:
        opt_config = yaml.safe_load(f) or {}

    spec = OptimizationSpec.from_yaml(opt_config_path, key="spec")

    # Base extraction schema (the thing being optimized)
    base_schema_path = Path(args.extraction_config)
    if not base_schema_path.exists():
        print(f"ERROR: extraction schema not found: {base_schema_path}", file=sys.stderr)
        return 2
    base_schema_yaml = base_schema_path.read_text()

    # Pipeline config (we only read OCR + processing settings here)
    pipeline_config: dict[str, Any] = {}
    pipeline_path = Path(args.pipeline_config)
    if pipeline_path.exists():
        with open(pipeline_path) as f:
            pipeline_config = yaml.safe_load(f) or {}

    # Merge: optimization.yaml LLM section overrides pipeline.yaml LLM section
    merged_config = {
        "llm": opt_config.get("llm") or pipeline_config.get("llm") or {},
        "ocr": opt_config.get("ocr") or pipeline_config.get("ocr") or {},
        "processing": opt_config.get("processing") or pipeline_config.get("processing") or {},
    }

    # Dataset
    examples = load_dataset(args.dataset_dir)
    if len(examples) == 0:
        print("ERROR: dataset is empty", file=sys.stderr)
        return 2
    dev_set, train_set, test_set = split_dataset(examples)
    logger.info(f"Dataset: {len(examples)} examples → "
                f"dev={len(dev_set)}, train={len(train_set)}, test={len(test_set)}")

    # Initialize DB and create run
    init_engine()
    run_id = repo.create_run(
        name=spec.name, spec_dict=spec.to_dict(), base_schema_yaml=base_schema_yaml,
    )
    logger.info(f"Created run: {run_id}")

    # Build root agent
    started_at = time.time()
    root_agent = build_optimizer_root_agent(
        run_id=run_id,
        spec=spec,
        config=merged_config,
        strategy_descriptor_dir=args.strategy_dir or "config/strategies",
        prompt_template_path=args.prompt_template or "prompts/extraction_prompt.yaml",
        started_at_unix=started_at,
    )

    # Initial session state
    initial_state = {
        K.RUN_ID: run_id,
        K.RUN_SPEC: spec.to_dict(),
        K.RUN_BASE_SCHEMA_YAML: base_schema_yaml,
        K.RUN_DEV_SET: dev_set,
        K.RUN_TRAIN_SET: train_set,
        K.RUN_TEST_SET: test_set,
        K.RALPH_ITERATION: 0,
        K.RALPH_HISTORY: [],
        K.GEPA_ITERATION: 0,
        K.GEPA_ARCHIVE: [],
        K.GEPA_DEV_RESULTS: {},
        K.GEPA_STAGNANT_ROUNDS: 0,
        K.GEPA_CURRENT_SCHEMA_YAML: base_schema_yaml,
    }

    # Run via ADK Runner with DatabaseSessionService
    final_status = asyncio.run(_run_optimizer(
        run_id=run_id, root_agent=root_agent, initial_state=initial_state,
        db_url_for_adk=args.adk_db_url,
    ))

    # Finish run with appropriate status
    archive = repo.latest_archive(run_id)
    final_cand = None
    if archive:
        # Pick the best by mean_field_f1
        best = max(archive, key=lambda e: e["objectives"].get("mean_field_f1", 0))
        final_cand = best["candidate_id"]

    repo.finish_run(run_id, status=final_status, final_candidate_id=final_cand)
    print(f"\nRun complete: {run_id}")
    print(f"Status: {final_status}")
    if final_cand:
        cand = repo.get_candidate(final_cand)
        print(f"Best candidate: {final_cand}")
        if cand and cand.get("objectives"):
            for k, v in cand["objectives"].items():
                print(f"  {k}: {v:.3f}")
    print(f"\nNext: python -m optimizer.cli promote {run_id} {final_cand or '<cand_id>'}")
    return 0


async def _run_optimizer(
    run_id: str,
    root_agent,
    initial_state: dict,
    db_url_for_adk: str | None,
) -> str:
    """Run the agent graph via the ADK Runner."""
    from google.adk.runners import Runner
    from google.adk.sessions import DatabaseSessionService, InMemorySessionService
    from google.genai import types as gtypes

    if db_url_for_adk:
        session_service = DatabaseSessionService(db_url=db_url_for_adk)
    else:
        # Fallback: in-memory (state still persists to OUR optimizer tables)
        logger.warning("No --adk-db-url provided; using InMemorySessionService for ADK. "
                       "ADK session state will not survive pod restarts.")
        session_service = InMemorySessionService()

    user_id = "optimizer"
    app_name = "doc-extract-optimizer"
    session = await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=run_id, state=initial_state,
    )

    runner = Runner(
        agent=root_agent, app_name=app_name, session_service=session_service,
    )

    user_message = gtypes.Content(
        role="user", parts=[gtypes.Part(text=f"Start optimization run {run_id}")],
    )

    final_status = "exhausted"
    try:
        async for event in runner.run_async(
            user_id=user_id, session_id=run_id, new_message=user_message,
        ):
            if event.actions and event.actions.escalate:
                # Inner escalations don't end the run; only top-level loop end does
                pass
        # Reload session and inspect spec satisfaction
        final_session = await session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=run_id,
        )
        if final_session and final_session.state.get(K.RALPH_SHOULD_STOP):
            spec_dict = final_session.state.get(K.RUN_SPEC, {}) or {}
            archive = final_session.state.get(K.GEPA_ARCHIVE, []) or []
            metrics = {}
            if archive:
                best = max(archive, key=lambda e: e["objectives"].get("mean_field_f1", 0))
                metrics = best["objectives"]
            from optimizer.spec import Budget, OptimizationSpec, TargetMetric
            spec = OptimizationSpec(
                name=spec_dict.get("name", ""),
                description=spec_dict.get("description", ""),
                targets=[TargetMetric(**t) for t in spec_dict.get("targets", [])],
                budget=Budget(**(spec_dict.get("budget") or {})),
            )
            satisfied, _ = spec.evaluate_satisfaction(metrics)
            final_status = "satisfied" if satisfied else "exhausted"
    except Exception as e:
        logger.exception(f"Run failed: {e}")
        final_status = "failed"

    return final_status


# ============================================================================
# status command
# ============================================================================

def cmd_status(args) -> int:
    init_engine()
    run = repo.get_run(args.run_id)
    if not run:
        print(f"Run not found: {args.run_id}", file=sys.stderr)
        return 2

    archive = repo.latest_archive(args.run_id)
    costs = repo.cost_totals(args.run_id)

    print(f"Run: {run['id']}")
    print(f"  Name:     {run['name']}")
    print(f"  Status:   {run['status']}")
    print(f"  Started:  {run['started_at']}")
    print(f"  Finished: {run['finished_at'] or '(running)'}")
    print(f"\nCosts:")
    print(f"  Tokens:  {costs['total_tokens']:,}")
    print(f"  Dollars: ${costs['estimated_dollars']:.2f}")
    print(f"\nPareto frontier: {len(archive)} candidate(s)")
    for entry in archive:
        obj = entry["objectives"] or {}
        print(f"  {entry['candidate_id']}:")
        for k, v in obj.items():
            print(f"    {k}: {v:.3f}" if isinstance(v, (int, float)) else f"    {k}: {v}")

    if run["final_candidate_id"]:
        print(f"\nFinal candidate: {run['final_candidate_id']}")

    return 0


# ============================================================================
# promote command
# ============================================================================

def cmd_promote(args) -> int:
    init_engine()
    cand = repo.get_candidate(args.candidate_id)
    if not cand:
        print(f"Candidate not found: {args.candidate_id}", file=sys.stderr)
        return 2
    if cand["run_id"] != args.run_id:
        print(f"Candidate {args.candidate_id} does not belong to run {args.run_id}", file=sys.stderr)
        return 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cand["schema_yaml"])
    print(f"Promoted: wrote {len(cand['schema_yaml'])} bytes to {output_path}")
    if cand.get("objectives"):
        print(f"Candidate metrics:")
        for k, v in cand["objectives"].items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.3f}")
    return 0


# ============================================================================
# argparse
# ============================================================================

def main():
    p = argparse.ArgumentParser(prog="optimizer", description="Ralph+GEPA prompt optimizer")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start", help="Start a new optimization run")
    p_start.add_argument("--optimization-config", default="config/optimization.yaml",
                         help="Path to optimization.yaml")
    p_start.add_argument("--extraction-config", default="config/extraction_schema.yaml",
                         help="Path to base extraction schema")
    p_start.add_argument("--pipeline-config", default="config/pipeline.yaml",
                         help="Path to pipeline.yaml (OCR / processing settings)")
    p_start.add_argument("--prompt-template", default="prompts/extraction_prompt.yaml",
                         help="Path to extraction prompt template")
    p_start.add_argument("--strategy-dir", default="config/strategies",
                         help="Directory of strategy descriptor YAMLs")
    p_start.add_argument("--dataset-dir", default="dataset",
                         help="Dataset directory (with documents/ and expected/)")
    p_start.add_argument("--adk-db-url", default=None,
                         help="DB URL for ADK SessionService (defaults to OPTIMIZER_DATABASE_URL)")
    p_start.set_defaults(func=cmd_start)

    p_status = sub.add_parser("status", help="Show run status")
    p_status.add_argument("run_id")
    p_status.set_defaults(func=cmd_status)

    p_promote = sub.add_parser("promote", help="Write a candidate's schema to disk")
    p_promote.add_argument("run_id")
    p_promote.add_argument("candidate_id")
    p_promote.add_argument("--output", default="config/extraction_schema_optimized.yaml",
                           help="Where to write the promoted schema")
    p_promote.set_defaults(func=cmd_promote)

    args = p.parse_args()
    if args.cmd == "start" and not getattr(args, "adk_db_url", None):
        # If env var present, pass it through
        import os
        args.adk_db_url = os.environ.get("OPTIMIZER_DATABASE_URL")
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
