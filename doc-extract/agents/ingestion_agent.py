"""Ingestion agent - downloads document via StorageBackend, extracts text."""

from __future__ import annotations

import logging

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from core.document_extractor import DocumentExtractor
from core.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class IngestionAgent(BaseAgent):
    """Reads `source_location` and `source_key` from session state, writes
    `document_text`, `page_count`, `ocr_used`, `extraction_method` back."""

    def __init__(
        self,
        name: str,
        storage: StorageBackend,
        extractor: DocumentExtractor,
    ):
        super().__init__(name=name)
        self._storage = storage
        self._extractor = extractor

    async def _run_async_impl(self, ctx: InvocationContext):
        state = ctx.session.state
        location = state["source_location"]
        key = state["source_key"]

        uri = self._storage.describe(location, key)
        logger.info(f"[{self.name}] ingesting {uri}")

        content = self._storage.download_to_bytes(location, key)
        extracted = self._extractor.extract(content, filename=key)

        state_delta = {
            "document_text": extracted.text,
            "page_count": extracted.page_count,
            "ocr_used": extracted.ocr_used,
            "extraction_method": extracted.extraction_method,
            "document_size_bytes": len(content),
        }

        msg = (
            f"Ingested {key}: {extracted.page_count} pages, "
            f"{len(extracted.text)} chars, "
            f"method={extracted.extraction_method}"
        )

        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=msg)]),
            actions=EventActions(state_delta=state_delta),
        )
