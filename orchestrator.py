"""Orchestrator - assembles the full Ralph+GEPA agent graph.

Topology produced:

    ralph_loop = LoopAgent
      ├── strategy_selector
      ├── ralph_iteration_tracker
      ├── gepa_loop = LoopAgent
      │     ├── gepa_iteration_tracker
      │     ├── reflection_agent
      │     ├── proposal_agent
      │     ├── eval_fanout = ParallelAgent
      │     │     ├── candidate_evaluator_0
      │     │     ├── ...
      │     │     └── candidate_evaluator_{K-1}
      │     ├── pareto_archive_agent
      │     └── gepa_convergence_check  <- escalates inner loop
      └── spec_validator                 <- escalates outer loop

Plus a one-shot bootstrap_eval agent that runs once outside the ralph_loop
to evaluate the base schema and seed the archive.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from google.adk.agents import LoopAgent, ParallelAgent, SequentialAgent

from core.document_extractor import DocumentExtractor
from optimizer.agents import (
    BootstrapEvalAgent, CandidateEvaluatorAgent, GEPAConvergenceCheck,
    GEPAIterationTracker, ParetoArchiveAgent, ProposalAgent,
    RalphIterationTracker, ReflectionAgent, SpecValidator,
    StrategySelectorAgent,
)
from optimizer.llm_client import (
    CostTrackingCallable, LLMCallable, make_llm,
)
from optimizer.persistence import repo
from optimizer.spec import OptimizationSpec
from optimizer.strategies import list_strategies

logger = logging.getLogger(__name__)


def build_optimizer_root_agent(
    run_id: str,
    spec: OptimizationSpec,
    config: dict[str, Any],
    strategy_descriptor_dir: str | Path,
    prompt_template_path: str | Path,
    started_at_unix: float,
) -> SequentialAgent:
    """Build the full optimizer agent graph.

    Returns a SequentialAgent whose children are:
      [bootstrap_eval, ralph_loop]
    """
    K_candidates = spec.candidates_per_gepa_iteration

    # ---------- LLMs ----------
    extraction_provider = config.get("llm", {}).get("extraction", {}).get("provider", "gemini")
    extraction_model = config.get("llm", {}).get("extraction", {}).get("model", "gemini-2.0-flash-exp")
    extraction_temp = config.get("llm", {}).get("extraction", {}).get("temperature", 0.0)

    proposer_provider = config.get("llm", {}).get("optimizer", {}).get("provider", extraction_provider)
    proposer_model = config.get("llm", {}).get("optimizer", {}).get("model", extraction_model)
    proposer_temp = config.get("llm", {}).get("optimizer", {}).get("temperature", 0.7)

    judge_provider = config.get("llm", {}).get("judge", {}).get("provider", extraction_provider)
    judge_model = config.get("llm", {}).get("judge", {}).get("model", extraction_model)
    judge_temp = config.get("llm", {}).get("judge", {}).get("temperature", 0.0)

    extraction_llm_raw = make_llm(extraction_provider, extraction_model, extraction_temp)
    proposer_llm_raw = make_llm(proposer_provider, proposer_model, proposer_temp)
    judge_llm_raw = make_llm(judge_provider, judge_model, judge_temp)

    # Wrap each with cost tracking
    def _budget_check(rid: str = run_id):
        return repo.cost_totals(rid)

    extraction_llm = CostTrackingCallable(
        extraction_llm_raw, run_id, "extraction",
        spec.budget.max_dollars, _budget_check,
    )
    proposer_llm = CostTrackingCallable(
        proposer_llm_raw, run_id, "proposer",
        spec.budget.max_dollars, _budget_check,
    )
    judge_llm = CostTrackingCallable(
        judge_llm_raw, run_id, "judge",
        spec.budget.max_dollars, _budget_check,
    )

    # ---------- Document extractor ----------
    ocr_cfg = config.get("ocr", {}) or {}
    document_extractor = DocumentExtractor(
        strategy=ocr_cfg.get("strategy", "native_first"),
        text_threshold_chars_per_page=ocr_cfg.get("text_threshold_chars_per_page", 50),
        ocr_engine=ocr_cfg.get("engine", "docling"),
    )

    # ---------- Build ParallelAgent of K candidate evaluators ----------
    eval_branches = [
        CandidateEvaluatorAgent(
            name=f"candidate_evaluator_{i}",
            candidate_slot=i,
            extraction_llm=extraction_llm,
            judge_llm=judge_llm,
            document_extractor=document_extractor,
            prompt_template_path=str(prompt_template_path),
            max_parallel_docs=config.get("processing", {}).get("max_parallel_docs", 4),
        )
        for i in range(K_candidates)
    ]
    eval_fanout = ParallelAgent(
        name="eval_fanout",
        sub_agents=eval_branches,
        description=f"Parallel evaluation of {K_candidates} candidates",
    )

    # ---------- GEPA inner loop ----------
    gepa_loop = LoopAgent(
        name="gepa_loop",
        max_iterations=spec.gepa_max_iterations,
        sub_agents=[
            GEPAIterationTracker(name="gepa_iteration_tracker"),
            ReflectionAgent(name="reflection_agent", llm_fn=proposer_llm),
            ProposalAgent(
                name="proposal_agent",
                strategy_descriptor_dir=strategy_descriptor_dir,
                llm_fn=proposer_llm,
            ),
            eval_fanout,
            ParetoArchiveAgent(name="pareto_archive", n_slots=K_candidates),
            GEPAConvergenceCheck(
                name="gepa_convergence",
                max_iterations=spec.gepa_max_iterations,
                patience=spec.convergence_patience,
            ),
        ],
        description="GEPA inner optimization loop",
    )

    # ---------- Ralph outer loop ----------
    available_strategies = list_strategies()
    ralph_loop = LoopAgent(
        name="ralph_loop",
        max_iterations=spec.ralph_max_iterations,
        sub_agents=[
            StrategySelectorAgent(
                name="strategy_selector",
                available_strategies=available_strategies,
            ),
            RalphIterationTracker(name="ralph_iteration_tracker"),
            gepa_loop,
            SpecValidator(name="spec_validator", run_started_at_unix=started_at_unix),
        ],
        description="Ralph outer loop with strategy selection",
    )

    # ---------- Bootstrap evaluator (one-shot) ----------
    bootstrap_evaluator = CandidateEvaluatorAgent(
        name="bootstrap_evaluator_inner",
        candidate_slot=0,
        extraction_llm=extraction_llm,
        judge_llm=judge_llm,
        document_extractor=document_extractor,
        prompt_template_path=str(prompt_template_path),
        max_parallel_docs=config.get("processing", {}).get("max_parallel_docs", 4),
    )
    bootstrap_eval = BootstrapEvalAgent(
        name="bootstrap_eval",
        evaluator=bootstrap_evaluator,
    )

    # ---------- Top-level wrapper ----------
    return SequentialAgent(
        name="prompt_optimizer",
        sub_agents=[bootstrap_eval, ralph_loop],
        description="Ralph(GEPA) prompt optimizer with three-signal scoring",
    )
