"""Pipeline orchestrator - composes the SequentialAgent from configured stages."""

from __future__ import annotations

import logging

from google.adk.agents import SequentialAgent

from agents.extraction_agent import ExtractionAgent
from agents.ingestion_agent import IngestionAgent
from agents.output_agent import OutputAgent
from agents.validation_agent import ValidationAgent
from core.document_extractor import DocumentExtractor
from core.prompt_builder import PromptBuilder
from core.schema import ExtractionSchema, PipelineConfig
from core.storage import build_storage
from core.storage.base import StorageBackend

logger = logging.getLogger(__name__)


def build_pipeline(
    config: PipelineConfig,
    schema: ExtractionSchema,
    prompt_template_path: str,
    storage: StorageBackend | None = None,
) -> tuple[SequentialAgent, StorageBackend]:
    """Build the full extraction pipeline. Returns (pipeline, storage)."""

    if storage is None:
        storage = build_storage(config.get("storage", default={}) or {})

    ocr_cfg = config.get("ocr", default={}) or {}
    extractor = DocumentExtractor(
        strategy=ocr_cfg.get("strategy", "native_first"),
        text_threshold_chars_per_page=ocr_cfg.get("text_threshold_chars_per_page", 50),
        ocr_engine=ocr_cfg.get("engine", "docling"),
        ocr_config=ocr_cfg.get("docling", {}),
    )

    prompt_builder = PromptBuilder(schema=schema, template_path=prompt_template_path)

    ingestion = IngestionAgent(name="ingestion", storage=storage, extractor=extractor)

    extraction = ExtractionAgent(
        name="extraction",
        model=config.get("llm", "extraction", "model", default="gemini-2.0-flash-exp"),
        prompt_builder=prompt_builder,
        use_chain_of_thought=config.get("extraction", "use_chain_of_thought", default=True),
        max_retries=config.get("extraction", "max_retries", default=2),
        temperature=config.get("llm", "extraction", "temperature", default=0.0),
    )

    validation = ValidationAgent(name="validation", schema=schema)

    output = OutputAgent(
        name="output",
        storage=storage,
        output_location=config.get("storage", "output_location"),
        output_prefix=config.get("storage", "output_prefix", default=""),
        include_raw_text=config.get("output", "include_raw_text", default=False),
        include_metadata=config.get("output", "include_extraction_metadata", default=True),
    )

    pipeline = SequentialAgent(
        name="document_extraction_pipeline",
        sub_agents=[ingestion, extraction, validation, output],
        description=f"Extract structured data from {schema.document_type} documents",
    )
    return pipeline, storage
