"""Validation agent - enforces schema and business rules on extracted data.

Pure Python validation - no LLM call, deterministic and fast.
Writes `validation_errors` (list[dict]) and `validation_status` to state.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from core.schema import ExtractionSchema

logger = logging.getLogger(__name__)


class ValidationAgent(BaseAgent):
    def __init__(self, name: str, schema: ExtractionSchema):
        super().__init__(name=name)
        self._schema = schema

    async def _run_async_impl(self, ctx: InvocationContext):
        state = ctx.session.state
        data = state.get("extracted_data")

        errors: list[dict] = []

        if data is None:
            errors.append({
                "rule": "extraction_present",
                "severity": "error",
                "message": "No extracted_data available - extraction step failed",
            })
            yield self._emit(errors)
            return

        # Required fields
        for fd in self._schema.fields:
            if fd.required and (fd.name not in data or data[fd.name] in (None, "", [])):
                errors.append({
                    "rule": "required_field",
                    "field": fd.name,
                    "severity": "error",
                    "message": f"Required field '{fd.name}' missing or empty",
                })

        # Type and enum validation
        for fd in self._schema.fields:
            if fd.name not in data or data[fd.name] is None:
                continue
            val = data[fd.name]
            type_err = self._check_type(fd.name, val, fd.type, fd.format)
            if type_err:
                errors.append(type_err)
            if fd.enum and isinstance(val, str) and val not in fd.enum:
                errors.append({
                    "rule": "enum",
                    "field": fd.name,
                    "severity": "error",
                    "message": f"Value '{val}' not in allowed enum {fd.enum}",
                })

        # Custom validation rules
        for rule in self._schema.validation_rules:
            err = self._apply_rule(rule, data)
            if err:
                errors.append(err)

        yield self._emit(errors)

    def _emit(self, errors: list[dict]) -> Event:
        has_errors = any(e["severity"] == "error" for e in errors)
        status = "failed" if has_errors else ("warning" if errors else "passed")
        return Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=f"Validation: {status} ({len(errors)} issues)")]),
            actions=EventActions(state_delta={
                "validation_errors": errors,
                "validation_status": status,
            }),
        )

    # ----- Type checks -----

    def _check_type(self, name: str, val: Any, expected_type: str, fmt: str | None) -> dict | None:
        if expected_type in ("string",):
            if not isinstance(val, str):
                return self._type_error(name, val, expected_type)
        elif expected_type == "number":
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                return self._type_error(name, val, expected_type)
        elif expected_type == "integer":
            if not isinstance(val, int) or isinstance(val, bool):
                return self._type_error(name, val, expected_type)
        elif expected_type == "boolean":
            if not isinstance(val, bool):
                return self._type_error(name, val, expected_type)
        elif expected_type == "date":
            if not isinstance(val, str):
                return self._type_error(name, val, expected_type)
            try:
                datetime.strptime(val, "%Y-%m-%d")
            except ValueError:
                return {
                    "rule": "date_format",
                    "field": name,
                    "severity": "error",
                    "message": f"Date '{val}' does not match expected format YYYY-MM-DD",
                }
        elif expected_type == "array":
            if not isinstance(val, list):
                return self._type_error(name, val, expected_type)
        return None

    def _type_error(self, name: str, val: Any, expected: str) -> dict:
        return {
            "rule": "type",
            "field": name,
            "severity": "error",
            "message": f"Field '{name}' has type {type(val).__name__}, expected {expected}",
        }

    # ----- Custom rules -----

    def _apply_rule(self, rule, data: dict) -> dict | None:
        # Built-in rules by name. Add new ones here as needed.
        try:
            if rule.name == "line_item_total_check":
                items = data.get("line_items") or []
                subtotal = data.get("subtotal")
                if items and subtotal is not None:
                    item_sum = sum(float(it.get("total", 0) or 0) for it in items)
                    if abs(item_sum - float(subtotal)) > rule.tolerance:
                        return self._rule_err(rule, f"sum(line_items.total)={item_sum} != subtotal={subtotal}")
            elif rule.name == "total_arithmetic_check":
                sub = data.get("subtotal")
                tax = data.get("tax_amount") or 0
                tot = data.get("total_amount")
                if sub is not None and tot is not None:
                    expected = float(sub) + float(tax)
                    if abs(expected - float(tot)) > rule.tolerance:
                        return self._rule_err(rule, f"subtotal+tax={expected} != total={tot}")
            elif rule.name == "due_date_after_invoice_date":
                inv = data.get("invoice_date")
                due = data.get("due_date")
                if inv and due:
                    inv_d = datetime.strptime(inv, "%Y-%m-%d")
                    due_d = datetime.strptime(due, "%Y-%m-%d")
                    if due_d < inv_d:
                        return self._rule_err(rule, f"due_date {due} before invoice_date {inv}")
        except (ValueError, TypeError) as e:
            return self._rule_err(rule, f"rule evaluation failed: {e}")
        return None

    def _rule_err(self, rule, msg: str) -> dict:
        return {
            "rule": rule.name,
            "severity": rule.severity,
            "message": f"{rule.description}: {msg}",
        }
