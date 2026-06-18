from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from .scenario_retrieval import RetrievalBundle
from .safety_lens import SafetyLensV2Result


@dataclass(frozen=True)
class LlmAssistResult:
    scene_summary: str
    hara_reasoning: str
    severity_rationale: str
    exposure_rationale: str
    controllability_rationale: str
    asil_hint: str
    recommended_follow_up_tests: list[str]
    raw_response: dict[str, Any] | None = None
    error: str | None = None


def build_llm_assist_payload(
    scenario_name: str,
    scenario_tags: list[str],
    safety_result: SafetyLensV2Result,
    retrieval_bundle: RetrievalBundle,
    metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "scenario": {
            "name": scenario_name or "Unspecified scene",
            "tags": scenario_tags,
        },
        "deterministic_layer": {
            "severity": safety_result.severity,
            "expected_objects": safety_result.expected_objects,
            "detected_objects": safety_result.detected_objects,
            "low_confidence_expected_objects": safety_result.low_confidence_expected_objects,
            "missed_expected_objects": safety_result.missed_expected_objects,
            "evidence_chain": safety_result.evidence_chain,
            "metrics": {
                "precision": None if metrics is None else metrics.get("precision"),
                "recall": None if metrics is None else metrics.get("recall"),
                "map50": None if metrics is None else metrics.get("map50"),
                "map50_95": None if metrics is None else metrics.get("map50_95"),
                "display_threshold": None if metrics is None else metrics.get("display_threshold"),
                "low_confidence_threshold": None if metrics is None else metrics.get("low_confidence_threshold"),
            },
        },
        "scenario_retrieval_layer": {
            "similar_known_scenarios": [
                {
                    "title": item.title,
                    "source": item.source_path.name,
                    "matched_terms": item.matched_terms,
                    "excerpt": item.excerpt,
                }
                for item in retrieval_bundle.similar_scenarios
            ],
            "relevant_safety_context": [
                {
                    "title": item.title,
                    "source": item.source_path.name,
                    "excerpt": item.excerpt,
                }
                for item in retrieval_bundle.safety_context
            ],
            "project1_standards_guidance": [
                {
                    "title": item.title,
                    "source": item.source_path.name,
                    "excerpt": item.excerpt,
                }
                for item in retrieval_bundle.standards_guidance
            ],
        },
    }


def _build_prompt(payload: dict[str, Any]) -> str:
    return (
        "You are a safety-engineering assistant. Use the provided deterministic evidence and retrieved "
        "Project 1 safety context to write a concise HARA-style assist. "
        "Do not change metrics, counts, thresholds, or severity. "
        "Do not invent detections or standards. "
        "Return JSON only with this schema:\n"
        "{\n"
        '  "scene_summary": string,\n'
        '  "hara_reasoning": string,\n'
        '  "severity_rationale": string,\n'
        '  "exposure_rationale": string,\n'
        '  "controllability_rationale": string,\n'
        '  "asil_hint": string,\n'
        '  "recommended_follow_up_tests": [string, ...]\n'
        "}\n\n"
        "Use 'ASIL hint' language, not final ASIL assignment. "
        "Reference the retrieved scenarios and standards guidance where useful.\n\n"
        f"Payload:\n{json.dumps(payload, indent=2)}"
    )


def generate_local_llm_assist(
    payload: dict[str, Any],
    model_name: str,
    base_url: str = "http://localhost:11434",
    timeout_seconds: int = 45,
) -> LlmAssistResult:
    prompt = _build_prompt(payload)
    body = json.dumps(
        {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
    ).encode("utf-8")

    req = request.Request(
        url=f"{base_url.rstrip('/')}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        return LlmAssistResult(
            scene_summary="",
            hara_reasoning="",
            severity_rationale="",
            exposure_rationale="",
            controllability_rationale="",
            asil_hint="",
            recommended_follow_up_tests=[],
            error=f"Could not reach local LLM service at {base_url}: {exc}",
        )
    except Exception as exc:
        return LlmAssistResult(
            scene_summary="",
            hara_reasoning="",
            severity_rationale="",
            exposure_rationale="",
            controllability_rationale="",
            asil_hint="",
            recommended_follow_up_tests=[],
            error=f"Local LLM request failed: {exc}",
        )

    raw_text = data.get("response", "").strip()
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return LlmAssistResult(
            scene_summary="",
            hara_reasoning="",
            severity_rationale="",
            exposure_rationale="",
            controllability_rationale="",
            asil_hint="",
            recommended_follow_up_tests=[],
            raw_response={"ollama": data, "raw_text": raw_text},
            error=f"Local LLM returned non-JSON output: {exc}",
        )

    return LlmAssistResult(
        scene_summary=str(parsed.get("scene_summary", "")),
        hara_reasoning=str(parsed.get("hara_reasoning", "")),
        severity_rationale=str(parsed.get("severity_rationale", "")),
        exposure_rationale=str(parsed.get("exposure_rationale", "")),
        controllability_rationale=str(parsed.get("controllability_rationale", "")),
        asil_hint=str(parsed.get("asil_hint", "")),
        recommended_follow_up_tests=[str(item) for item in parsed.get("recommended_follow_up_tests", [])],
        raw_response=parsed,
        error=None,
    )


def render_llm_assist_markdown(result: LlmAssistResult) -> str:
    if result.error:
        return "\n".join(
            [
                "### LLM Assist Layer",
                "",
                f"- Error: {result.error}",
            ]
        )

    lines = [
        "### LLM Assist Layer",
        "",
        "#### Scene Summary",
        result.scene_summary or "No summary returned.",
        "",
        "#### HARA-style Reasoning Draft",
        result.hara_reasoning or "No reasoning draft returned.",
        "",
        "#### Candidate Severity Rationale",
        result.severity_rationale or "No severity rationale returned.",
        "",
        "#### Candidate Exposure Rationale",
        result.exposure_rationale or "No exposure rationale returned.",
        "",
        "#### Candidate Controllability Rationale",
        result.controllability_rationale or "No controllability rationale returned.",
        "",
        "#### Proposed ASIL Hint",
        result.asil_hint or "No ASIL hint returned.",
        "",
        "#### Recommended Follow-up Tests",
    ]

    if result.recommended_follow_up_tests:
        lines.extend(f"- {item}" for item in result.recommended_follow_up_tests)
    else:
        lines.append("- No follow-up tests returned.")

    return "\n".join(lines)
