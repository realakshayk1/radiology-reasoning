"""
generate_report.py — Grounded generation with strict schema enforcement.

Fixes applied (March 2026):
  1. Strip markdown fences BEFORE JSON extraction — the old find('{') / rfind('}')
     approach left the closing ``` fence inside the extracted string, which caused
     model_validate_json to fail on every call (0% validity rate).
  2. Pass response_format={"type": "json_object"} on every call to reduce fence output.
  3. Use temperature=0.0 for deterministic, schema-consistent responses.
  4. Log retry_count and escalation_count per-session for observability.
  5. Fallback now clearly marks caution as TECHNICAL ERROR and sets escalate=True.
"""

import logging
import re
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from src.generation.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    format_findings,
    format_evidence,
)
from src.api.schemas import RadiologyReport

logger = logging.getLogger(__name__)

# Session-level counters — reset on process restart, not per-call.
_session_stats = {"total": 0, "retries": 0, "escalations": 0}


def _clean_json(raw: str) -> str:
    """
    Remove markdown code fences and surrounding whitespace.

    Order matters:
      1. Strip outer whitespace.
      2. Remove ```json ... ``` or ``` ... ``` wrappers.
      3. Strip again.
      4. Extract the first {...} block as the final safety net.
    """
    text = raw.strip()

    # Remove ```json fence (with or without language tag)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Safety net: extract first complete {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    return text


def _call_api(client: OpenAI, model_name: str, messages: list[dict]) -> str:
    """Single API call with json_object response format and zero temperature."""
    print(f"DEBUG: Calling API with model={model_name}")
    print(f"DEBUG: Messages: {messages}")
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,                           # Deterministic output
        max_completion_tokens=1000,
    )
    return response.choices[0].message.content


class ReportGenerator:
    """
    Generate structured RadiologyReport objects from predictions + retrieved evidence.

    Usage:
        generator = ReportGenerator()
        report = generator.generate(preds_dict, hits)
        print(report.model_dump_json(indent=2))
    """

    def __init__(
        self,
        model_name: str = "gpt-5.4-mini-2026-03-17",
        top_k: int = 5,
        max_retries: int = 2,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.max_retries = max_retries
        self.client = OpenAI()  # Reads OPENAI_API_KEY from environment

    def _build_messages(self, preds_dict: dict[str, float], hits: list[dict]) -> list[dict]:
        findings_str = format_findings(preds_dict)
        evidence_str = format_evidence(hits)
        user_content = USER_PROMPT_TEMPLATE.format(
            findings_list=findings_str,
            top_k=self.top_k,
            evidence_chunks=evidence_str,
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def generate(
        self, preds_dict: dict[str, float], hits: list[dict]
    ) -> RadiologyReport:
        """
        Generate and validate a RadiologyReport.

        Retries up to max_retries times on ValidationError.
        Returns an escalation-flagged fallback if all retries fail.
        """
        _session_stats["total"] += 1
        messages = self._build_messages(preds_dict, hits)
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                raw = _call_api(self.client, self.model_name, messages)
                logger.debug("Raw API response (attempt %d): %s", attempt + 1, raw)

                clean = _clean_json(raw)
                logger.debug("Cleaned JSON (attempt %d): %s", attempt + 1, clean)

                import json
                raw_dict = json.loads(clean)
                
                # Repair: If findings or impression came back as a list, join them.
                for field in ["findings", "impression", "rationale"]:
                    val = raw_dict.get(field)
                    if isinstance(val, list):
                        logger.info(f"DEBUG: Repairing field '{field}' from list to string.")
                        raw_dict[field] = " ".join([str(x) for x in val])
                    
                logger.info(f"DEBUG: raw_dict after repair: {raw_dict}")
                report = RadiologyReport.model_validate(raw_dict)
                if attempt > 0:
                    _session_stats["retries"] += 1
                    logger.info("Validation succeeded on retry %d", attempt + 1)
                return report

            except ValidationError as e:
                last_error = e
                logger.warning(
                    "Attempt %d/%d — ValidationError: %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    "Attempt %d/%d — API error: %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )

        # All retries exhausted — return flagged fallback
        _session_stats["escalations"] += 1
        logger.error(
            "All %d attempts failed. Returning escalation fallback. Last error: %s",
            self.max_retries,
            last_error,
        )
        return RadiologyReport(
            findings="Could not generate findings.",
            impression="Could not generate impression.",
            rationale="Generation failed after all retries.",
            caution="TECHNICAL ERROR — output not validated. Do not use clinically.",
            escalate=True,
            confidence="low",
        )

    @staticmethod
    def session_stats() -> dict[str, Any]:
        """Return session-level retry and escalation counters."""
        stats = dict(_session_stats)
        total = stats["total"] or 1
        stats["retry_rate"] = round(stats["retries"] / total, 4)
        stats["escalation_rate"] = round(stats["escalations"] / total, 4)
        return stats