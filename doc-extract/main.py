"""Console application entrypoint.

Lists documents from configured storage (local folder or S3), runs each
through the ADK pipeline, writes results back.

Usage:
  python main.py                              # process all docs from input_location
  python main.py --key path/to/doc.pdf        # process single doc
  python main.py --limit 5                    # process first 5
  python main.py --schema config/other.yaml   # different schema
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

sys.path.insert(0, str(Path(__file__).parent))

from agents.pipeline import build_pipeline
from core.schema import ExtractionSchema, PipelineConfig

APP_NAME = "doc_extraction"
USER_ID = "system"


def setup_logging(level: str = "INFO", structured: bool = False):
    fmt = (
        '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
        if structured
        else "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout)


async def process_one(
    runner: Runner,
    session_service: InMemorySessionService,
    location: str,
    key: str,
) -> dict:
    log = logging.getLogger("process_one")
    session_id = f"sess-{uuid.uuid4().hex[:12]}"

    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
        state={"source_location": location, "source_key": key},
    )

    trigger = types.Content(
        role="user",
        parts=[types.Part(text=f"Extract document {key}")],
    )

    try:
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=trigger,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        log.info(f"[{event.author}] {part.text[:200]}")

        final = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
        state = final.state if final else {}

        return {
            "key": key,
            "status": state.get("validation_status", "unknown"),
            "extraction_status": state.get("extraction_status"),
            "errors": len(state.get("validation_errors", [])),
            "output_key": state.get("output_key"),
            "ocr_used": state.get("ocr_used"),
        }
    except Exception as e:
        log.exception(f"Pipeline failed for {key}: {e}")
        return {"key": key, "status": "exception", "error": str(e)}


async def run(args: argparse.Namespace):
    log = logging.getLogger("main")

    config = PipelineConfig.from_yaml(args.pipeline_config)
    schema = ExtractionSchema.from_yaml(args.schema)

    setup_logging(
        level=config.get("logging", "level", default="INFO"),
        structured=config.get("logging", "structured", default=False),
    )

    log.info(f"Loaded schema: {schema.document_type} ({len(schema.fields)} fields)")
    log.info(f"Storage backend: {config.get('storage', 'backend', default='local')}")

    pipeline, storage = build_pipeline(
        config=config,
        schema=schema,
        prompt_template_path=args.prompt_template,
    )

    session_service = InMemorySessionService()
    runner = Runner(agent=pipeline, app_name=APP_NAME, session_service=session_service)

    input_location = config.get("storage", "input_location")
    output_location = config.get("storage", "output_location")
    input_prefix = config.get("storage", "input_prefix", default="")

    if not input_location or not output_location:
        log.error("Storage input_location / output_location not configured")
        sys.exit(2)

    if args.key:
        keys = [args.key]
    else:
        keys = [obj.key for obj in storage.list_documents(input_location, prefix=input_prefix)]
        if args.limit:
            keys = keys[: args.limit]

    log.info(f"Processing {len(keys)} document(s) from {input_location}/{input_prefix}")

    max_concurrent = config.get("processing", "max_concurrent_documents", default=5)
    sem = asyncio.Semaphore(max_concurrent)

    async def _bounded(key: str):
        async with sem:
            return await process_one(runner, session_service, input_location, key)

    results = await asyncio.gather(*[_bounded(k) for k in keys])

    log.info("=" * 60)
    log.info("EXTRACTION SUMMARY")
    log.info("=" * 60)
    passed = sum(1 for r in results if r["status"] == "passed")
    warned = sum(1 for r in results if r["status"] == "warning")
    failed = sum(1 for r in results if r["status"] not in ("passed", "warning"))

    for r in results:
        log.info(f"  {r['status']:10} {r['key']}  errors={r.get('errors', '?')}")

    log.info("-" * 60)
    log.info(f"Total: {len(results)}  passed={passed}  warning={warned}  failed={failed}")
    log.info(f"Outputs in {output_location}/{config.get('storage', 'output_prefix', default='')}")

    return 0 if failed == 0 else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Document extraction console app")
    p.add_argument("--schema", default="config/extraction_schema.yaml")
    p.add_argument("--pipeline-config", default="config/pipeline.yaml")
    p.add_argument("--prompt-template", default="prompts/extraction_prompt.yaml")
    p.add_argument("--key", help="Process a single key instead of listing")
    p.add_argument("--limit", type=int, help="Process at most N documents")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(asyncio.run(run(args)))
