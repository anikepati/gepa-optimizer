"""Extraction agent - the core LLM call that produces structured output.

Designed to support multiple LLM backends (Gemini native, Claude/OpenAI via
LiteLlm) and to use structured output / tool calling where the backend supports
it for guaranteed schema conformance.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from core.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class ExtractionAgent(LlmAgent):
    """LlmAgent specialized for structured extraction.

    Reads `document_text` from state, writes `extracted_data` (parsed JSON),
    `extraction_reasoning` (the CoT scratchpad), and `raw_llm_response`.
    """

    def __init__(
        self,
        name: str,
        model: str,
        prompt_builder: PromptBuilder,
        use_chain_of_thought: bool = True,
        max_retries: int = 2,
        temperature: float = 0.0,
    ):
        # Build the extraction instruction. ADK's LlmAgent uses `instruction`
        # as the system prompt and feeds the conversation as user messages.
        # We construct the full prompt at runtime via _run_async_impl override
        # so we can inject the document into the user message.
        super().__init__(
            name=name,
            model=model,
            instruction=prompt_builder.build_system_prompt(),
            output_key="raw_llm_response",
            generate_content_config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=4096,
            ),
        )
        self._prompt_builder = prompt_builder
        self._use_cot = use_chain_of_thought
        self._max_retries = max_retries

    async def _run_async_impl(self, ctx: InvocationContext):
        state = ctx.session.state
        document_text = state.get("document_text", "")
        if not document_text:
            yield Event(
                author=self.name,
                content=types.Content(parts=[types.Part(text="ERROR: no document_text")]),
                actions=EventActions(escalate=True),
            )
            return

        # Build the extraction user prompt
        user_prompt = self._prompt_builder.build_extraction_prompt(
            document_text=document_text,
            use_chain_of_thought=self._use_cot,
        )

        # Inject as a user turn for this invocation only.
        # We do this by appending to ctx.session events, then calling super().
        from google.adk.events import Event as ADKEvent
        ctx.session.events.append(
            ADKEvent(
                author="user",
                content=types.Content(
                    role="user",
                    parts=[types.Part(text=user_prompt)],
                ),
            )
        )

        # Let the parent LlmAgent stream the response
        last_text = ""
        async for event in super()._run_async_impl(ctx):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        last_text += part.text
            yield event

        # Parse the response into reasoning + structured output
        reasoning, parsed = self._parse_response(last_text)

        state_delta: dict[str, Any] = {
            "raw_llm_response": last_text,
            "extraction_reasoning": reasoning,
        }

        if parsed is not None:
            state_delta["extracted_data"] = parsed
            state_delta["extraction_status"] = "success"
        else:
            state_delta["extraction_status"] = "parse_error"
            state_delta["extracted_data"] = None

        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=f"Extraction status: {state_delta['extraction_status']}")]),
            actions=EventActions(state_delta=state_delta),
        )

    # ----- Response parsing -----

    def _parse_response(self, text: str) -> tuple[str, dict | None]:
        """Pull <reasoning> and <output> sections, fall back to bare JSON."""
        reasoning = ""
        m_reason = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
        if m_reason:
            reasoning = m_reason.group(1).strip()

        # Prefer explicit <output> tag
        m_out = re.search(r"<output>(.*?)</output>", text, re.DOTALL)
        json_str = m_out.group(1).strip() if m_out else text

        # Strip markdown fences if present
        json_str = re.sub(r"^```(?:json)?\s*", "", json_str.strip())
        json_str = re.sub(r"\s*```$", "", json_str)

        # Try to find a JSON object substring if there's surrounding prose
        try:
            return reasoning, json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Fallback: scan for first {...} block
        brace_match = re.search(r"\{.*\}", json_str, re.DOTALL)
        if brace_match:
            try:
                return reasoning, json.loads(brace_match.group(0))
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse failed: {e}")

        return reasoning, None
