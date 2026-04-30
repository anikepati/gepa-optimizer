"""Output agent - assembles final result envelope and writes via StorageBackend."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from core.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class OutputAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        storage: StorageBackend,
        output_location: str,
        output_prefix: str,
        include_raw_text: bool = False,
        include_metadata: bool = True,
    ):
        super().__init__(name=name)
        self._storage = storage
        self._output_location = output_location
        self._output_prefix = (output_prefix.rstrip("/") + "/") if output_prefix else ""
        self._include_raw_text = include_raw_text
        self._include_metadata = include_metadata

    async def _run_async_impl(self, ctx: InvocationContext):
        state = ctx.session.state

        source_key = state["source_key"]
        source_location = state["source_location"]

        stem = Path(source_key).with_suffix(".json").name
        output_key = f"{self._output_prefix}{stem}"

        envelope: dict[str, Any] = {
            "source": {"location": source_location, "key": source_key},
            "extraction": state.get("extracted_data"),
            "validation": {
                "status": state.get("validation_status"),
                "errors": state.get("validation_errors", []),
            },
        }

        if self._include_metadata:
            envelope["metadata"] = {
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "page_count": state.get("page_count"),
                "ocr_used": state.get("ocr_used"),
                "extraction_method": state.get("extraction_method"),
                "document_size_bytes": state.get("document_size_bytes"),
                "extraction_status": state.get("extraction_status"),
                "reasoning": state.get("extraction_reasoning"),
            }

        if self._include_raw_text:
            envelope["raw_text"] = state.get("document_text")

        self._storage.upload_json(envelope, self._output_location, output_key)
        uri = self._storage.describe(self._output_location, output_key)

        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=f"Wrote result to {uri}")]),
            actions=EventActions(state_delta={"output_key": output_key}),
        )
